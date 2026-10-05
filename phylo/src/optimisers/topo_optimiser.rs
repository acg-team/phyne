use std::fmt::Display;

use itertools::Itertools;
use log::{debug, info};
use rand::{Rng, SeedableRng};

use crate::alignment::{Alignment, AncestralAlignment};
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{
    BranchOptimiser, MoveCostInfo, MoveOptimiser, NniOptimiser, SprOptimiser, StopCondition,
    TreeOptimisationResult,
};
use crate::parsimony::scoring::ParsimonyScoring;
use crate::parsimony::{BasicParsimonyCost, DolloParsimonyCost};
use crate::pip_model::PIPCost;
use crate::random::RandomGenerator;
use crate::substitution_models::{QMatrix, SubstitutionCost};
use crate::tkf_model::{TKFCost, TKFIndelCost, TKFModel as TKFM};
use crate::tree::NodeIdx;
use crate::Result;

/// The `Compatible` trait is used to ensure that the cost and move optimiser passed to
/// [`TopologyOptimiser::new`] are compatible.
pub trait Compatible<MO: MoveOptimiser> {}

impl<Q: QMatrix, A: Alignment> Compatible<SprOptimiser> for PIPCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<NniOptimiser> for PIPCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<SprOptimiser> for SubstitutionCost<Q, A> {}
impl<Q: QMatrix, A: Alignment> Compatible<NniOptimiser> for SubstitutionCost<Q, A> {}
impl<S: ParsimonyScoring, A: Alignment> Compatible<SprOptimiser> for DolloParsimonyCost<S, A> {}
impl<S: ParsimonyScoring, A: Alignment> Compatible<NniOptimiser> for DolloParsimonyCost<S, A> {}
impl<A: Alignment> Compatible<SprOptimiser> for BasicParsimonyCost<A> {}
impl<A: Alignment> Compatible<NniOptimiser> for BasicParsimonyCost<A> {}
impl<T: TKFM, AA: AncestralAlignment> Compatible<NniOptimiser> for TKFIndelCost<T, AA> {}
impl<Q: QMatrix, T: TKFM, AA: AncestralAlignment> Compatible<NniOptimiser> for TKFCost<Q, T, AA> {}

pub struct TopologyOptimiser<'a, MO, C, R>
where
    MO: MoveOptimiser,
    C: TreeSearchCost + Display + Clone + Send + Compatible<MO>,
    R: Rng + SeedableRng,
{
    pub(crate) stop_condition: StopCondition,
    pub(crate) move_opti: MO,
    pub(crate) c: C,
    pub(crate) rng: &'a mut RandomGenerator<R>,
}

impl<'a, MO, C, R> TopologyOptimiser<'a, MO, C, R>
where
    MO: MoveOptimiser,
    C: TreeSearchCost + Display + Clone + Send + Compatible<MO>,
    R: Rng + SeedableRng,
{
    pub fn new(cost: C, move_opti: MO, rng: &'a mut RandomGenerator<R>) -> Self {
        Self {
            move_opti,
            c: cost,
            stop_condition: StopCondition::default(),
            rng,
        }
    }

    pub fn with_stop_condition(
        cost: C,
        move_opti: MO,
        rng: &'a mut RandomGenerator<R>,
        stop: StopCondition,
    ) -> Self {
        Self {
            c: cost,
            move_opti,
            stop_condition: stop,
            rng,
        }
    }

    /// Runs the topology optimisation algorithm on the given cost function given a move optimiser.
    /// The algorithm will iterate until the predicate is satisfied.
    /// The cost function will be updated in place.
    ///
    /// # Panics
    /// Panics if the provided tree move is not applicable to the tree, e.g. SPR move will panic
    /// if the tree has less than 4 nodes, as SPRs are not applicable.
    ///
    /// # Returns
    /// A `PhyloOptimisationResult` containing the initial cost, final cost, number of iterations, and the final cost function.
    /// The final cost function will contain the optimised tree.
    ///
    /// # Example
    /// ```rust
    /// use phylo::likelihood::TreeSearchCost;
    /// use phylo::optimisers::{SprOptimiser, TopologyOptimiser};
    /// use phylo::phylo_info::PhyloInfoBuilder;
    /// use phylo::random::DefaultGenerator;
    /// use phylo::substitution_models::{K80, SubstModel, SubstitutionCostBuilder};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let info = PhyloInfoBuilder::new("./examples/data/K80.fasta").build()?;
    /// let k80 = SubstModel::<K80>::default();
    /// let c = SubstitutionCostBuilder::new(k80, info).build()?;
    /// let unopt_cost = c.cost();
    /// let result = TopologyOptimiser::new(c, SprOptimiser {}, &mut DefaultGenerator::default()).run()?;
    /// assert_eq!(unopt_cost, result.initial_cost);
    /// assert!(result.final_cost > result.initial_cost);
    /// assert!(result.iterations <= 100);
    /// assert_eq!(result.cost.tree().len(), 9); // The initial tree has 9 nodes, 5 leaves and 4 internal nodes.
    /// # Ok(()) }
    /// ```
    pub fn run(mut self) -> Result<TreeOptimisationResult<C>> {
        info!("Optimising tree topology");
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

        info!("Done optimising tree topology");
        info!("Final cost: {curr_cost}, achieved in {iterations} iteration(s)");
        Ok(TreeOptimisationResult {
            initial_cost: init_cost,
            final_cost: curr_cost,
            iterations,
            costs,
            cost: self.c,
        })
    }

    /// Performs a single iteration of topology optimisation over all nodes/branches in the tree.
    /// Returns the cost after going through all possible moves once.
    fn single_optimisation_iteration(&mut self) -> Result<f64> {
        let init_cost = self.c.cost();
        let possible_move_locs: Vec<_> = self.move_opti.move_locations(&self.c).copied().collect();
        let mut current_move_locs: Vec<_> = possible_move_locs.iter().collect();

        self.rng.shuffle(&mut current_move_locs);
        let move_opti = self.move_opti.clone();

        let mut curr_cost =
            Self::fold_improving_moves(&mut self.c, &move_opti, init_cost, &current_move_locs)?;

        // Optimise branch lengths on current tree to match PhyML
        if self.c.blen_optimisation() {
            let intermediate_stop_condition = match self.stop_condition {
                StopCondition::Epsilon(e) => StopCondition::epsilon(e),
                StopCondition::MaxIterEpsilon(_, e) => StopCondition::epsilon(e),
                _ => StopCondition::default(),
            };

            let o =
                BranchOptimiser::with_stop_condition(self.c.clone(), intermediate_stop_condition)
                    .run()?;
            if o.final_cost > curr_cost {
                curr_cost = o.final_cost;
                let mut tree = o.cost.tree().clone();
                tree.dirty();
                self.c.update_tree(tree);
            }
        }

        Ok(curr_cost)
    }

    /// Iterates over `move_locations` in order and applies the best (improving)
    /// tree move for each move location in place
    /// # Returns:
    /// - the new cost (or `base_cost` if no improvement was found)
    pub fn fold_improving_moves(
        cost_fn: &mut C,
        move_opti: &MO,
        base_cost: f64,
        move_locations: &[&NodeIdx],
    ) -> Result<f64> {
        debug_assert!(
            {
                let correct_move_locations = move_opti.move_locations(cost_fn).collect_vec();
                move_locations
                    .iter()
                    .all(|move_location| correct_move_locations.contains(move_location))
            },
            "all move locations must be contained in the tree and valid"
        );

        move_locations.iter().copied().try_fold(
            base_cost,
            |base_cost, move_location| -> Result<_> {
                let move_cost_info =
                    move_opti.best_move_at_location(base_cost, cost_fn, move_location)?;
                let MoveCostInfo {
                    cost: best_cost,
                    tree: best_tree,
                } = move_cost_info;

                if best_cost > base_cost {
                    // TODO: can we not re-use the cost struct that was used to compute the
                    // best_move_at_location, instead of calling update_tree() again?
                    // For tkf we would have to call the dp re-estimate again here again, even
                    // though it is already computed once during the best_move_at_location call.
                    // See issue #141 https://github.com/acg-team/rust-phylo/issues/141
                    cost_fn.update_tree(best_tree);
                    info!("    {move_opti} move applied, new cost {best_cost}");
                    Ok(best_cost)
                } else {
                    info!("    No improvement, best cost {best_cost}");
                    Ok(base_cost)
                }
            },
        )
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
    use crate::random::FakeGenerator;
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
    fn single_iter_pip_template<
        Q: QMatrix + QMatrixMaker + Default + Send,
        MO: MoveOptimiser + Clone + Send + 'static,
    >(
        info: PhyloInfo<MSA>,
        move_optimiser: MO,
    ) where
        PIPCost<Q, MSA>: Compatible<MO>,
    {
        let mut rng = FakeGenerator::default();
        let model = PIPModel::<Q>::default();
        let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
        let init_cost = c.cost();

        let mut optimiser = TopologyOptimiser::new(c.clone(), move_optimiser, &mut rng);
        let optimised_cost = optimiser.single_optimisation_iteration().unwrap();

        assert!(optimised_cost > init_cost);
        assert_eq!(optimised_cost, optimiser.c.cost());

        // Check that branch lengths changed, topology should change because the tree is wrong on purpose
        let new_info = optimiser.c.info.clone();
        assert_ne!(new_info.tree.length, info.tree.length);
        assert_ne!(new_info.tree.robinson_foulds(&info.tree), 0);

        // Check that the cost is the same when recomputed from the new info and same model
        let new_cost = PIPCB::new(model, new_info).build().unwrap();
        assert_eq!(new_cost.cost(), optimised_cost);
    }

    #[test]
    fn single_iter_pip_dna_nni() {
        let info = dna_test_data();
        single_iter_pip_template::<JC69, NniOptimiser>(info, NniOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iter_pip_dna_nni_long() {
        let info = dna_test_data();
        single_iter_pip_template::<K80, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_pip_template::<HKY, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_pip_template::<TN93, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_pip_template::<GTR, NniOptimiser>(info, NniOptimiser {});
    }

    #[test]
    fn single_iter_pip_dna_spr() {
        let info = dna_test_data();
        single_iter_pip_template::<JC69, SprOptimiser>(info, SprOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iter_pip_dna_spr_long() {
        let info = dna_test_data();
        single_iter_pip_template::<K80, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_pip_template::<HKY, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_pip_template::<TN93, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_pip_template::<GTR, SprOptimiser>(info, SprOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_pip_aa_nni() {
        let info = aa_test_data();
        single_iter_pip_template::<WAG, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_pip_template::<BLOSUM, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_pip_template::<HIVB, NniOptimiser>(info, NniOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_pip_aa_spr() {
        let info = aa_test_data();
        single_iter_pip_template::<WAG, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_pip_template::<BLOSUM, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_pip_template::<HIVB, SprOptimiser>(info, SprOptimiser {});
    }

    #[cfg(test)]
    fn single_iter_substitution_template<
        Q: QMatrix + QMatrixMaker + Default + Send,
        MO: MoveOptimiser + Clone + Send + 'static,
    >(
        info: PhyloInfo<MSA>,
        move_optimiser: MO,
    ) where
        SubstitutionCost<Q, MSA>: Compatible<MO>,
    {
        let mut rng = FakeGenerator::default();
        let model = SubstModel::<Q>::default();
        let c = SCB::new(model.clone(), info.clone()).build().unwrap();
        let init_cost = c.cost();

        let mut optimiser = TopologyOptimiser::new(c.clone(), move_optimiser, &mut rng);
        let optimised_cost = optimiser.single_optimisation_iteration().unwrap();

        assert!(optimised_cost > init_cost);
        assert_eq!(optimised_cost, optimiser.c.cost());

        // Check that branch lengths changed, topology should change because the tree is wrong on purpose
        let new_info = optimiser.c.info.clone();
        assert_ne!(new_info.tree.length, info.tree.length);
        assert_ne!(new_info.tree.robinson_foulds(&info.tree), 0);

        // Check that the cost is the same when recomputed from the new info and same model
        let new_cost = SCB::new(model, new_info).build().unwrap();
        assert_eq!(new_cost.cost(), optimised_cost);
    }

    #[test]
    fn single_iteration_substitution_dna_nni() {
        let info = dna_test_data();
        single_iter_substitution_template::<JC69, NniOptimiser>(info, NniOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_substitution_dna_nni_long() {
        // This test takes too long for coverage runs
        let info = dna_test_data();
        single_iter_substitution_template::<K80, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_substitution_template::<HKY, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_substitution_template::<TN93, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_substitution_template::<GTR, NniOptimiser>(info, NniOptimiser {});
    }

    #[test]
    fn single_iteration_substitution_dna_spr() {
        let info = dna_test_data();
        single_iter_substitution_template::<JC69, SprOptimiser>(info, SprOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_substitution_dna_spr_long() {
        let info = dna_test_data();
        single_iter_substitution_template::<K80, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_substitution_template::<HKY, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_substitution_template::<TN93, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_substitution_template::<GTR, SprOptimiser>(info, SprOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_substitution_aa_nni() {
        let info = aa_test_data();
        single_iter_substitution_template::<WAG, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_substitution_template::<BLOSUM, NniOptimiser>(info.clone(), NniOptimiser {});
        single_iter_substitution_template::<HIVB, NniOptimiser>(info.clone(), NniOptimiser {});
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn single_iteration_substitution_aa_spr() {
        let info = aa_test_data();
        single_iter_substitution_template::<WAG, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_substitution_template::<BLOSUM, SprOptimiser>(info.clone(), SprOptimiser {});
        single_iter_substitution_template::<HIVB, SprOptimiser>(info.clone(), SprOptimiser {});
    }
}
