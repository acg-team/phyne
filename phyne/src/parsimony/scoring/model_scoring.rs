use std::fmt::Display;

use log::{debug, info};
use ordered_float::OrderedFloat;

use crate::alphabets::ParsimonySet;
use crate::evolutionary_models::EvoModel;
use crate::parsimony::{
    CostMatrix, DiagonalZeros, GapCost, ParsimonyModel, ParsimonyScoring, Rounding,
};
use crate::Result;

#[derive(Clone, Debug, PartialEq)]
pub struct ModelScoringBuilder<P: ParsimonyModel> {
    model: P,
    gap: GapCost,
    diagonal: DiagonalZeros,
    rounding: Rounding,
    times: Vec<f64>,
}

impl<P: ParsimonyModel + EvoModel> ModelScoringBuilder<P> {
    pub fn new(model: P) -> Self {
        ModelScoringBuilder {
            model,
            gap: GapCost::new(2.5, 1.0),
            diagonal: DiagonalZeros::non_zero(),
            rounding: Rounding::none(),
            times: Vec::new(),
        }
    }

    pub fn gap_cost(mut self, gap: GapCost) -> Self {
        self.gap = gap;
        self
    }

    pub fn diagonal(mut self, diagonal: DiagonalZeros) -> Self {
        self.diagonal = diagonal;
        self
    }

    pub fn rounding(mut self, rounding: Rounding) -> Self {
        self.rounding = rounding;
        self
    }

    pub fn times(mut self, times: Vec<f64>) -> Self {
        self.times = times;
        self
    }

    pub fn build(self) -> Result<ModelScoring<P>> {
        info!(
            "Setting up the parsimony scoring from the {} model",
            self.model
        );

        let mut times = self
            .times
            .iter()
            .cloned()
            .map(OrderedFloat::<f64>::from)
            .collect::<Vec<_>>();
        times.sort();

        let costs: Vec<(OrderedFloat<f64>, TimeCosts)> = times
            .iter()
            .map(|time| {
                let cost_matrix =
                    self.model
                        .scoring(f64::from(*time), &self.diagonal, &self.rounding);
                let avg = cost_matrix.mean();
                (
                    *time,
                    TimeCosts {
                        avg,
                        gap: self.gap * avg,
                        c: cost_matrix,
                    },
                )
            })
            .collect();

        info!(
            "Created scoring matrices from the {} substitution model for {times:?} branch lengths",
            self.model
        );
        debug!("The scoring matrices are: {costs:?}");
        Ok(ModelScoring {
            model: self.model,
            gap: self.gap,
            costs,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ModelScoring<P: ParsimonyModel> {
    model: P,
    gap: GapCost,
    costs: Vec<(OrderedFloat<f64>, TimeCosts)>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TimeCosts {
    avg: f64,
    gap: GapCost,
    c: CostMatrix,
}

impl<P: ParsimonyModel> Display for ModelScoring<P> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Model-based parsimony scoring with {}, {}",
            self.model, self.gap
        )
    }
}

impl<P: ParsimonyModel> ModelScoring<P> {
    fn scoring(&self, target: OrderedFloat<f64>) -> &TimeCosts {
        let target = OrderedFloat(target);
        match self.costs.binary_search_by(|(time, _)| time.cmp(&target)) {
            Ok(idx) => &self.costs[idx].1, // Exact match
            Err(idx) => {
                if idx == 0 {
                    &self.costs[0].1
                } else if idx == self.costs.len() {
                    &self.costs[self.costs.len() - 1].1
                } else {
                    // Pick closest value
                    let (before, _) = self.costs[idx - 1];
                    let (after, _) = self.costs[idx];
                    if (target.0 - before).abs() <= (after - target.0).abs() {
                        &self.costs[idx - 1].1
                    } else {
                        &self.costs[idx].1
                    }
                }
            }
        }
    }
}

impl<P: ParsimonyModel + EvoModel> ParsimonyScoring for ModelScoring<P> {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64 {
        self.scoring(OrderedFloat(blen)).c[(P::alphabet().index(i), P::alphabet().index(j))]
    }

    fn min_match(&self, blen: f64, i: &ParsimonySet, j: &ParsimonySet) -> f64 {
        // Find minimal score among all pairs of chars in the sets (all to all)
        i.iter()
            .flat_map(|&a| j.iter().map(move |&b| self.r#match(blen, &a, &b)))
            .fold(f64::INFINITY, f64::min)
    }

    fn gap_open(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).gap.open
    }

    fn gap_ext(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).gap.ext
    }

    fn avg(&self, blen: f64) -> f64 {
        self.scoring(OrderedFloat(blen)).avg
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use std::fmt::Debug;

    use approx::assert_relative_eq;
    use ordered_float::OrderedFloat;

    use crate::parsimony::{
        CostMatrix, DiagonalZeros as Z, GapCost, ModelScoringBuilder, ModelScoringBuilder as MCB,
        ParsimonyModel, ParsimonyScoring, Rounding as R,
    };
    use crate::substitution_models::*;

    const TRUE_COST_MATRIX: [f64; 400] = [
        0.0, 6.0, 6.0, 5.0, 6.0, 6.0, 5.0, 4.0, 7.0, 7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 4.0, 4.0, 9.0,
        7.0, 4.0, 5.0, 0.0, 6.0, 7.0, 7.0, 5.0, 6.0, 5.0, 5.0, 7.0, 5.0, 3.0, 7.0, 8.0, 6.0, 5.0,
        6.0, 6.0, 7.0, 6.0, 5.0, 6.0, 0.0, 4.0, 8.0, 5.0, 5.0, 5.0, 5.0, 6.0, 7.0, 4.0, 8.0, 8.0,
        7.0, 4.0, 4.0, 9.0, 6.0, 6.0, 5.0, 7.0, 4.0, 0.0, 9.0, 6.0, 3.0, 5.0, 6.0, 8.0, 7.0, 6.0,
        8.0, 8.0, 6.0, 5.0, 6.0, 9.0, 7.0, 7.0, 5.0, 6.0, 7.0, 8.0, 0.0, 8.0, 8.0, 6.0, 7.0, 7.0,
        6.0, 7.0, 7.0, 6.0, 7.0, 5.0, 6.0, 7.0, 6.0, 5.0, 5.0, 4.0, 5.0, 6.0, 8.0, 0.0, 4.0, 6.0,
        5.0, 7.0, 5.0, 4.0, 6.0, 8.0, 5.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 6.0, 6.0, 3.0, 9.0, 4.0,
        0.0, 5.0, 6.0, 7.0, 7.0, 4.0, 7.0, 8.0, 6.0, 5.0, 5.0, 8.0, 7.0, 5.0, 4.0, 6.0, 5.0, 5.0,
        7.0, 7.0, 6.0, 0.0, 7.0, 8.0, 7.0, 6.0, 8.0, 8.0, 7.0, 5.0, 6.0, 8.0, 8.0, 7.0, 6.0, 5.0,
        4.0, 5.0, 8.0, 4.0, 6.0, 6.0, 0.0, 7.0, 5.0, 5.0, 7.0, 6.0, 6.0, 5.0, 6.0, 8.0, 4.0, 7.0,
        6.0, 7.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 0.0, 4.0, 6.0, 5.0, 5.0, 8.0, 6.0, 5.0, 8.0,
        6.0, 3.0, 6.0, 6.0, 7.0, 8.0, 7.0, 6.0, 7.0, 7.0, 7.0, 4.0, 0.0, 6.0, 5.0, 5.0, 6.0, 6.0,
        6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 4.0, 6.0, 9.0, 4.0, 4.0, 6.0, 6.0, 6.0, 6.0, 0.0, 6.0, 8.0,
        6.0, 5.0, 5.0, 8.0, 8.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 4.0, 3.0, 5.0,
        0.0, 5.0, 7.0, 6.0, 5.0, 7.0, 6.0, 4.0, 6.0, 8.0, 8.0, 8.0, 7.0, 8.0, 8.0, 8.0, 6.0, 5.0,
        4.0, 7.0, 6.0, 0.0, 7.0, 6.0, 7.0, 6.0, 4.0, 5.0, 4.0, 6.0, 7.0, 6.0, 8.0, 6.0, 6.0, 6.0,
        6.0, 7.0, 6.0, 6.0, 8.0, 7.0, 0.0, 5.0, 5.0, 8.0, 7.0, 6.0, 4.0, 5.0, 4.0, 5.0, 6.0, 6.0,
        5.0, 5.0, 6.0, 6.0, 6.0, 5.0, 7.0, 6.0, 5.0, 0.0, 4.0, 7.0, 6.0, 6.0, 4.0, 6.0, 5.0, 6.0,
        7.0, 6.0, 5.0, 6.0, 7.0, 5.0, 6.0, 5.0, 6.0, 7.0, 6.0, 4.0, 0.0, 9.0, 7.0, 5.0, 7.0, 5.0,
        8.0, 7.0, 7.0, 7.0, 7.0, 6.0, 7.0, 7.0, 5.0, 7.0, 7.0, 5.0, 7.0, 6.0, 7.0, 0.0, 5.0, 6.0,
        6.0, 6.0, 5.0, 6.0, 7.0, 7.0, 7.0, 7.0, 5.0, 6.0, 6.0, 7.0, 7.0, 4.0, 7.0, 5.0, 6.0, 6.0,
        0.0, 6.0, 4.0, 7.0, 7.0, 7.0, 6.0, 7.0, 6.0, 6.0, 8.0, 3.0, 4.0, 6.0, 6.0, 6.0, 6.0, 6.0,
        5.0, 8.0, 7.0, 0.0,
    ];

    #[cfg(test)]
    fn builder_template<P: ParsimonyModel + Clone + PartialEq + Debug>(
        model: P,
        diagonal: Z,
        rounding: R,
    ) {
        let gap = GapCost::new(6.5, 3.0);
        let builder = ModelScoringBuilder::new(model.clone())
            .gap_cost(gap)
            .diagonal(diagonal)
            .rounding(rounding)
            .times(vec![0.1, 0.5, 10.0]);

        assert_eq!(builder.model, model);
        assert_eq!(builder.gap, gap);
        assert_eq!(builder.diagonal, diagonal);
        assert_eq!(builder.rounding, rounding);
    }

    #[test]
    fn model_builder_setters_protein() {
        for d in [Z::zero(), Z::non_zero()] {
            for r in [R::zero(), R::four(), R::none()] {
                builder_template(SubstModel::<HIVB>::default(), d, r);
                builder_template(SubstModel::<WAG>::default(), d, r);
                builder_template(SubstModel::<BLOSUM>::default(), d, r);
            }
        }
    }

    #[test]
    fn model_builder_setters_dna() {
        for d in [Z::zero(), Z::non_zero()] {
            for r in [R::zero(), R::four(), R::none()] {
                builder_template(SubstModel::<JC69>::default(), d, r);
                builder_template(SubstModel::<K80>::default(), d, r);
                builder_template(SubstModel::<HKY>::default(), d, r);
                builder_template(SubstModel::<TN93>::default(), d, r);
                builder_template(SubstModel::<GTR>::default(), d, r);
            }
        }
    }

    #[cfg(test)]
    fn builder_default_template<P: ParsimonyModel + Clone + PartialEq + Debug>(model: P) {
        use crate::parsimony::scoring::GapCost;

        let builder = ModelScoringBuilder::new(model.clone());

        assert_eq!(builder.model, model);
        assert_eq!(builder.gap, GapCost::new(2.5, 1.0));
        assert_eq!(builder.diagonal, Z::non_zero());
        assert_eq!(builder.rounding, R::none());
        assert_eq!(builder.times, Vec::<f64>::new());
    }

    #[test]
    fn model_builder_defaults_protein() {
        builder_default_template(SubstModel::<HIVB>::default());
        builder_default_template(SubstModel::<WAG>::default());
        builder_default_template(SubstModel::<BLOSUM>::default());
    }

    #[test]
    fn model_builder_defaults_dna() {
        builder_default_template(SubstModel::<JC69>::default());
        builder_default_template(SubstModel::<K80>::default());
        builder_default_template(SubstModel::<HKY>::default());
        builder_default_template(SubstModel::<TN93>::default());
        builder_default_template(SubstModel::<GTR>::default());
    }

    #[test]
    fn protein_scorings() {
        let s = MCB::new(SubstModel::<WAG>::default())
            .gap_cost(GapCost::new(2.5, 1.0))
            .diagonal(Z::non_zero())
            .rounding(R::zero())
            .times(vec![0.1, 0.3, 0.5, 0.7])
            .build()
            .unwrap();

        let mat_01 = s.scoring(OrderedFloat(0.1));
        let true_matrix_01 = CostMatrix::from_row_slice(20, 20, &TRUE_COST_MATRIX);
        assert_relative_eq!(mat_01.c, true_matrix_01);
        assert_relative_eq!(mat_01.avg, 5.7675);
        assert_relative_eq!(s.avg(0.3), 4.7475);
        assert_relative_eq!(s.avg(0.5), 4.2825);
        assert_relative_eq!(s.avg(0.7), 4.0075);
    }

    #[cfg(test)]
    fn rounding_template<P: ParsimonyModel + Clone>(model: P) {
        let gap = GapCost::new(2.5, 1.0);
        let s_rounded = MCB::new(model.clone())
            .gap_cost(gap)
            .diagonal(Z::zero())
            .rounding(R::zero())
            .times(vec![0.1])
            .build()
            .unwrap();
        let s = MCB::new(model)
            .gap_cost(gap)
            .diagonal(Z::zero())
            .times(vec![0.1])
            .build()
            .unwrap();

        assert_ne!(s_rounded.avg(0.1), s.avg(0.1));
        let rounded_costs = s_rounded.scoring(OrderedFloat(0.1));
        let costs = s.scoring(OrderedFloat(0.1));

        assert_ne!(rounded_costs.c, costs.c);
        for (&e1, &e2) in rounded_costs.c.iter().zip(costs.c.iter()) {
            assert_relative_eq!(e1, e1.round());
            assert_relative_eq!(e1, e2.round());
        }
    }

    #[test]
    fn matrix_entry_rounding_protein() {
        rounding_template(SubstModel::<HIVB>::default());
        rounding_template(SubstModel::<WAG>::default());
        rounding_template(SubstModel::<BLOSUM>::default());
    }

    #[test]
    fn matrix_entry_rounding_dna() {
        rounding_template(SubstModel::<JC69>::default());
        rounding_template(SubstModel::<K80>::default());
        rounding_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap());
        rounding_template(
            SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[442.29, 217.81]).unwrap(),
        );
        rounding_template(
            SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.0, 0.2, 0.2, 0.2, 0.2]).unwrap(),
        );
    }

    #[cfg(test)]
    fn matrix_zero_diagonals_template<P: ParsimonyModel + Clone>(model: P) {
        let gap = GapCost::new(2.5, 1.0);
        let s_zero_diags = MCB::new(model.clone())
            .gap_cost(gap)
            .diagonal(Z::zero())
            .times(vec![0.1])
            .build()
            .unwrap();

        let s = MCB::new(model)
            .gap_cost(gap)
            .times(vec![0.1])
            .build()
            .unwrap();

        assert_ne!(s_zero_diags.avg(0.1), s.avg(0.1));
        assert!(s_zero_diags.avg(0.1) < s.avg(0.1));

        let zero_diag_costs = s_zero_diags.scoring(OrderedFloat(0.1));
        let costs = s.scoring(OrderedFloat(0.1));
        assert_ne!(zero_diag_costs, costs);
        assert_eq!(zero_diag_costs.c.diagonal().iter().sum::<f64>(), 0.0);
        for (&e1, &e2) in zero_diag_costs
            .c
            .diagonal()
            .iter()
            .zip(costs.c.diagonal().iter())
        {
            assert_eq!(e1, 0.0);
            assert_ne!(e1, e2);
            assert_ne!(e2, 0.0);
        }
    }

    #[test]
    fn matrix_zero_diagonals_protein() {
        matrix_zero_diagonals_template(SubstModel::<HIVB>::default());
        matrix_zero_diagonals_template(SubstModel::<WAG>::default());
        matrix_zero_diagonals_template(SubstModel::<BLOSUM>::default());
    }

    #[test]
    fn matrix_zero_diagonals_dna() {
        matrix_zero_diagonals_template(SubstModel::<JC69>::default());
        matrix_zero_diagonals_template(SubstModel::<K80>::default());
        matrix_zero_diagonals_template(SubstModel::<HKY>::default());
        matrix_zero_diagonals_template(SubstModel::<TN93>::default());
        matrix_zero_diagonals_template(SubstModel::<GTR>::default());
    }
}
