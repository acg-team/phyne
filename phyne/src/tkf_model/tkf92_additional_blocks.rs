use std::cell::RefCell;
use std::fmt::Display;

use num_enum::FromPrimitive;

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::{
    blocks_of_alignment, merge_fragmentation_with_blocks, validate_fragmentation,
    validate_lambda_mu, validate_r, TKF92Parameters, TKFIndelCost, TKFIndelModelInfo, TKFModel,
};
use crate::Result;

/// [TKF92IndelModel](`super::TKF92IndelModel`) with additional block borders (and without a substitution model),
/// which means that the provided blocks will be used in addition to the blocks determined from
/// the alignment, see [`super::TKFModel::get_blocks`].
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct TKF92IndelModelAddBlocks {
    params: Vec<f64>,
    /// precomputed r.ln()
    ln_r: f64,
    /// precomputed ln((1 - r)/r)
    ln_one_minus_r_over_r: f64,
    /// Blocks to be used in addition to those determined from the alignment
    additional_blocks: Vec<usize>,
}

#[cfg(test)]
impl TKF92IndelModelAddBlocks {
    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

#[cfg(test)]
impl TKFModel for TKF92IndelModelAddBlocks {
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
            _ => {
                self.params[idx] = value;
            }
        };
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
        self.lambda().ln() + ln_beta + self.ln_one_minus_r_over_r
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        ln_tree_event_factor
            + (block_len as f64 - 1.0) * (ln_tree_event_factor.exp()).ln_1p()
            + (block_len as f64) * self.ln_r
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        let blocks = blocks_of_alignment(msa);
        merge_fragmentation_with_blocks(&blocks, &self.additional_blocks)
    }
}

#[cfg(test)]
impl Display for TKF92IndelModelAddBlocks {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}, and additional blocks = {:?}",
            self.lambda(),
            self.mu(),
            self.r(),
            self.additional_blocks
        )
    }
}

/// Builder for the cost using the [`TKF92IndelModelAddBlocks`].
pub struct TKF92IndelAddBlocksCostBuilder<AA: AncestralAlignment> {
    params: Vec<f64>,
    phylo: PhyloInfo<AA>,
    additional_blocks: Vec<usize>,
}

#[cfg(test)]
impl<AA: AncestralAlignment> TKF92IndelAddBlocksCostBuilder<AA> {
    pub fn new(params: &[f64], additional_blocks: Vec<usize>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            phylo,
            additional_blocks,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF92IndelModelAddBlocks, AA>> {
        let mut params = self.params;
        validate_lambda_mu(&mut params);
        validate_r(&mut params);
        let additional_blocks =
            validate_fragmentation(&self.additional_blocks, self.phylo.msa.len());
        let r = params[usize::from(TKF92Parameters::R)];
        let model = TKF92IndelModelAddBlocks {
            params,
            ln_r: r.ln(),
            ln_one_minus_r_over_r: ((1.0 - r) / r).ln(),
            additional_blocks,
        };
        let info = TKFIndelModelInfo::new(&model, &self.phylo);
        Ok(TKFIndelCost {
            model,
            phylo: self.phylo.clone(),
            model_info: RefCell::new(info),
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use approx::assert_relative_eq;

    use crate::alignment::{Sequences, MASA};
    use crate::tkf_model::TKF92FixedIndelCostBuilder;
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[test]
    #[should_panic]
    fn tkf92_param_range_invalid_index() {
        let model = TKF92IndelModelAddBlocks {
            params: vec![0.5, 1.0, 0.3],
            ln_r: 0.0, // cache filled with dummy since it is not needed here
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not needed here
            additional_blocks: vec![],
        };
        // Use an invalid index
        model.param_range(3);
    }

    #[test]
    fn tkf92_add_blocks_model_fmt() {
        let tkf_indel_model = TKF92IndelModelAddBlocks {
            params: vec![1.1, 2.0, 0.3],
            ln_r: 0.0,                  // cache filled with dummy since it is not printed
            ln_one_minus_r_over_r: 0.0, // cache filled with dummy since it is not printed
            additional_blocks: vec![1, 2],
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(
            fmt,
            "TKF92 with lambda = 1.1, mu = 2, r = 0.3, and additional blocks = [1, 2]"
        );
    }

    #[test]
    fn tkf92_add_blocks_indel_set_param() {
        let mut model = TKF92IndelModelAddBlocks {
            params: vec![1.0, 2.0, 0.3],
            ln_r: 0.0,                  // dummy
            ln_one_minus_r_over_r: 0.0, // dummy
            additional_blocks: vec![],  // dummy
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
        assert_eq!(model.ln_one_minus_r_over_r, (-new_r).ln_1p() - new_r.ln());
    }

    #[test]
    fn tkf_add_blocks_manual_integration_over_fragmentations() {
        // By manually summing over unobserved fragmentations (that confirm with the additionally
        // provided block borders) we can verify that this TKF92 model integrates over all possible
        // fragmentations that are consistent with the MSA and the additional block borders.
        let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
        let seqs = Sequences::new_unchecked(vec![
            record!("A0", b"AAB---DD"),
            record!("B1", b"-ARAAAWD"),
            record!("I1", b"AAA---AD"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };
        let lambda = 1.0;
        let mu = 1.1;
        let r = 0.5;
        let additional_blocks = vec![2, 4];

        let tkf92_cost = TKF92IndelAddBlocksCostBuilder::new(
            &[lambda, mu, r],
            additional_blocks,
            phylo_info.clone(),
        )
        .build()
        .unwrap();
        let cost = tkf92_cost.logl();

        let mut sum_over_fragmentations_cost = 0.0;

        let fragmentations = [vec![2, 4], vec![2, 4, 5], vec![2, 4, 7], vec![2, 4, 5, 7]];
        for fragmentation in fragmentations {
            let fragment_cost = TKF92FixedIndelCostBuilder::new(
                &[lambda, mu, r],
                fragmentation,
                phylo_info.clone(),
            )
            .build()
            .unwrap();
            sum_over_fragmentations_cost += fragment_cost.logl().exp();
        }
        sum_over_fragmentations_cost = sum_over_fragmentations_cost.ln();
        assert_relative_eq!(cost, sum_over_fragmentations_cost);
    }
}
