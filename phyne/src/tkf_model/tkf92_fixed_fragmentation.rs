use std::cell::RefCell;
use std::fmt::Display;

use log::warn;
use num_enum::FromPrimitive;
use rstest::rstest;

use crate::alignment::AncestralAlignment;
use crate::likelihood::{ParamRange, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE};
use crate::phylo_info::PhyloInfo;
use crate::tkf_model::{
    blocks_of_alignment, validate_lambda_mu, validate_r, TKF92Parameters, TKFIndelCost,
    TKFIndelModelInfo, TKFModel, DEFAULT_LAMBDA, DEFAULT_MU, DEFAULT_R,
};
use crate::Result;

/// TKF92 indel model with a `fixed fragmentation` (and without a substitution model),
/// which means that the provided fragmentation will be regarded as the true fragmentation.
/// This is different to the [TKF92IndelModel](`crate::tkf_model::TKF92IndelModel`), which integrates
/// over all possible fragmentations that confirm with the observed alignment.
#[cfg(test)]
#[derive(Clone, Debug, PartialEq)]
pub struct TKF92FixedIndelModel {
    pub(super) params: Vec<f64>,
    /// precomputed r.ln()
    pub(super) ln_r: f64,
    /// The fixed fragmentation to be used
    pub(super) fragmentation: Vec<usize>,
}

#[cfg(test)]
impl TKF92FixedIndelModel {
    pub fn r(&self) -> f64 {
        self.params[usize::from(TKF92Parameters::R)]
    }
}

#[cfg(test)]
impl TKFModel for TKF92FixedIndelModel {
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
        self.lambda().ln() - self.mu().ln()
    }

    fn ln_insertion_factor_at_non_root(&self, ln_beta: f64) -> f64 {
        self.lambda().ln() + ln_beta
    }

    fn block_prob(&self, ln_tree_event_factor: f64, block_len: usize) -> f64 {
        ln_tree_event_factor + (block_len as f64 - 1.0) * self.ln_r + (1.0 - self.r()).ln()
    }

    fn get_blocks<AA: AncestralAlignment>(&self, msa: &AA) -> Vec<usize> {
        let alignment_blocks = blocks_of_alignment(msa);
        merge_fragmentation_with_blocks(&self.fragmentation, &alignment_blocks)
    }
}

/// Merges the user defined fragmentation with the observed block borders in the MSA.
/// Assumes both inputs are sorted and within MSA length.
/// This is basically a union of the two sets and then returning the sorted result.
/// This implementations achieves a better run time than the naive approach.
#[cfg(test)]
pub(super) fn merge_fragmentation_with_blocks(
    fragmentation: &[usize],
    blocks: &[usize],
) -> Vec<usize> {
    let mut frag_iter = fragmentation.iter().peekable();
    let mut merged = Vec::new();
    for block in blocks.iter() {
        let mut next_block = false;
        while let Some(&frag) = frag_iter.peek() {
            if frag > block {
                warn!("Observed right border of block {block} in MSA not in fragmentation, adding it.");
                merged.push(*block);
                next_block = true;
                break;
            } else if frag == block {
                next_block = true;
                break;
            } else {
                frag_iter.next();
            }
        }
        if next_block {
            continue;
        } else {
            warn!("Observed right border of block {block} in MSA not in fragmentation, adding it.");
            merged.push(*block);
        }
    }
    for frag in fragmentation {
        merged.push(*frag);
    }
    merged.sort();
    merged
}

#[cfg(test)]
impl Display for TKF92FixedIndelModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TKF92 with lambda = {}, mu = {}, r = {}, and fixed fragmentation = {:?}",
            self.lambda(),
            self.mu(),
            self.r(),
            self.fragmentation,
        )
    }
}

/// Builder for the cost using the [`TKF92FixedIndelModel`].
#[cfg(test)]
pub struct TKF92FixedIndelCostBuilder<AA: AncestralAlignment> {
    params: Vec<f64>,
    fragmentation: Vec<usize>,
    phylo: PhyloInfo<AA>,
}

/// Removes duplicates and out-of-bounds entries.
#[cfg(test)]
pub(super) fn validate_fragmentation(fragmentation: &[usize], msa_len: usize) -> Vec<usize> {
    let mut fragmentation = fragmentation.to_vec();
    let original_len = fragmentation.len();
    if original_len == 0 {
        return vec![];
    }
    fragmentation.sort();
    // dedup() must be called after sorting
    fragmentation.dedup();
    let deduped_len = fragmentation.len();
    if deduped_len < original_len {
        warn!("Fragmentation had duplicate entries, which were removed.");
    }
    fragmentation.retain(|&x| x > 0 && x <= msa_len);
    let retained_len = fragmentation.len();
    if retained_len < deduped_len {
        warn!("Fragmentation had entries out of bounds (0, seq_len], which were removed.");
    }
    fragmentation.to_vec()
}

#[cfg(test)]
impl<AA: AncestralAlignment> TKF92FixedIndelCostBuilder<AA> {
    pub fn new(params: &[f64], fragmentation: Vec<usize>, phylo: PhyloInfo<AA>) -> Self {
        Self {
            params: params.to_vec(),
            fragmentation,
            phylo,
        }
    }

    pub fn build(self) -> Result<TKFIndelCost<TKF92FixedIndelModel, AA>> {
        let mut params = self.params;
        let lambda_id = usize::from(TKF92Parameters::Lambda);
        let mu_id = usize::from(TKF92Parameters::Mu);
        let r_id = usize::from(TKF92Parameters::R);
        if params.len() != 3 {
            warn!(
                "Expected 3 parameters for TKF92 model (lambda, mu, r), but got {}",
                params.len()
            );
            warn!("Falling back to default values");
            params.resize(3, 0.0);
            params[lambda_id] = DEFAULT_LAMBDA;
            params[mu_id] = DEFAULT_MU;
            params[r_id] = DEFAULT_R;
        } else {
            validate_lambda_mu(&mut params);
            validate_r(&mut params);
        }
        let fragmentation = validate_fragmentation(&self.fragmentation, self.phylo.msa.len());
        let r = params[r_id];
        let model = TKF92FixedIndelModel {
            params,
            ln_r: r.ln(),
            fragmentation,
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
    use std::path::Path;

    use approx::assert_relative_eq;

    use crate::alignment::{Sequences, MASA};
    use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder};
    use crate::tkf_model::TKF92IndelCostBuilder;
    use crate::{record_wo_desc as record, tree};

    use super::*;

    #[cfg(test)]
    fn naive_merge(set1: &[usize], set2: &[usize]) -> Vec<usize> {
        let mut merged: Vec<usize> = set1.to_vec();
        merged.extend(set2.iter().cloned());
        merged.sort();
        merged.dedup();
        merged
    }

    #[test]
    fn tkf_validate_fragmentation() {
        let fragmentation = vec![3, 19, 3, 4, 58, 13, 0, 1, 0, 3, 4, 15, 16];
        let msa_len = 15;
        let validated = validate_fragmentation(&fragmentation, msa_len);
        assert_eq!(validated, vec![1, 3, 4, 13, 15]);
    }

    #[rstest]
    #[case( vec![3, 5, 7, 10], vec![5, 10, 12], vec![3, 5, 7, 10, 12])]
    #[case( vec![3, 7, 10, 12], vec![5, 10, 12], vec![3, 5, 7, 10, 12])]
    #[case( vec![], vec![5, 10, 12], vec![5, 10, 12])]
    #[case( vec![1, 2, 3, 4], vec![1, 2, 3, 4], vec![1, 2, 3, 4])]
    #[case( vec![1, 2, 4], vec![1, 2, 3, 4], vec![1, 2, 3, 4])]
    fn tkf_merge_fragmentations_with_blocks(
        #[case] fragmentation: Vec<usize>,
        #[case] blocks: Vec<usize>,
        #[case] expected: Vec<usize>,
    ) {
        let merged = merge_fragmentation_with_blocks(&fragmentation, &blocks);
        assert_eq!(merged, expected);
        assert_eq!(merged, naive_merge(&fragmentation, &blocks));
    }

    #[test]
    fn tkf_manual_integration_over_fragmentations() {
        let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
        let seqs = Sequences::new_unchecked(vec![
            record!("A0", b"AAB---D"),
            record!("B1", b"-ARAAAW"),
            record!("I1", b"AAA---A"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };
        let lambda = 1.0;
        let mu = 1.1;
        let r = 0.5;

        let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo_info.clone())
            .build()
            .unwrap();
        let cost = tkf92_cost.logl();

        let mut sum_over_fragmentations_cost = 0.0;
        let fragmentations = [
            vec![1, 3, 6, 7],
            vec![1, 2, 3, 6, 7],
            vec![1, 3, 4, 6, 7],
            vec![1, 3, 5, 6, 7],
            vec![1, 3, 4, 5, 6, 7],
            vec![1, 2, 3, 4, 6, 7],
            vec![1, 2, 3, 5, 6, 7],
            vec![1, 2, 3, 4, 5, 6, 7],
        ];

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

    #[test]
    fn tkf_manual_integration_over_fragmentations_not_passing_all_fragments() {
        // By manually summing over unobserved fragmentations we can verify that
        // the TKF92 model integrates over all possible fragmentations.
        let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
        let seqs = Sequences::new_unchecked(vec![
            record!("A0", b"AAB---D"),
            record!("B1", b"-ARAAAW"),
            record!("I1", b"AAA---A"),
        ]);
        let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
        let phylo_info = PhyloInfo { msa, tree };
        let lambda = 1.0;
        let mu = 1.1;
        let r = 0.5;

        let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo_info.clone())
            .build()
            .unwrap();
        let cost = tkf92_cost.logl();

        let mut sum_over_fragmentations_cost = 0.0;
        // its not necessary to pass the fragments that are inferred from the MSA
        // i.e., one can omit the indices 1, 3, 6, 7 and still get the same result
        // see also the test `tkf_manual_integration_over_fragmentations`
        let fragmentations = [
            vec![1, 6, 7],
            vec![1, 2, 3],
            vec![1, 4, 6],
            vec![1, 3, 5, 6, 7],
            vec![4, 5, 6, 7],
            vec![1, 2, 4, 6, 7],
            vec![1, 2, 3, 5, 6, 7],
            vec![2, 4, 5],
        ];

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

    #[test]
    #[should_panic]
    fn tkf92_param_range_invalid_index() {
        let model = TKF92FixedIndelModel {
            params: vec![0.5, 1.0, 0.3],
            ln_r: 0.0, // cache filled with dummy since it is not needed here
            fragmentation: vec![],
        };
        // Use an invalid index
        model.param_range(3);
    }

    #[test]
    fn tkf92_fixed_model_fmt() {
        let tkf_indel_model = TKF92FixedIndelModel {
            params: vec![1.1, 2.0, 0.3],
            ln_r: 0.0, // cache filled with dummy since it is not printed
            fragmentation: vec![1, 2],
        };

        let fmt = format!("{}", tkf_indel_model);

        assert_eq!(
            fmt,
            "TKF92 with lambda = 1.1, mu = 2, r = 0.3, and fixed fragmentation = [1, 2]"
        );
    }

    #[test]
    #[cfg_attr(feature = "ci_coverage", ignore)]
    fn tkf_compare_to_simulation() {
        // This uses the MASA from a simulation under the TKF92 model given a tree and parameters.
        // Since it is a simulation, we know the true fragmentation. So we compute the ln-likelihood
        // using the fixed fragmentation and compare it to the ln-likelihood obtained from the
        // simulation. Note that, we do not remove non-emitting columns from the alignment,
        // since the simulation probability includes them.
        let dir = Path::new("data/tkf/fixed_fragments/");
        let sequence_file = dir.join("masa_dna.fasta");
        let tree_file = dir.join("tree.newick");
        let phylo_info = PhyloInfoBuilder::with_attrs(sequence_file, tree_file)
            .build_with_ancestors()
            .unwrap();

        let fragmentation = vec![
            4, 5, 6, 7, 11, 12, 13, 14, 16, 21, 22, 23, 26, 28, 29, 30, 31, 33, 40, 41, 47, 48, 54,
            55, 58, 61, 62, 63, 67, 69, 70, 71, 74, 75, 76, 78, 80, 83, 84, 85, 86, 88, 91, 93, 96,
            97, 98, 100, 104, 105, 108, 110, 111, 112, 113, 114, 118, 119, 120, 127, 129, 130, 135,
            136, 137, 138, 139, 140, 141, 143, 144, 147, 149, 151, 152, 154, 160, 163, 164, 166,
            168, 169, 170, 172, 174, 175, 176, 177, 180, 181, 183, 185, 187, 189, 190, 193, 195,
            198, 199, 200, 201, 202, 205, 206, 211, 212, 213, 219, 220, 222, 223, 224, 226, 227,
            228, 231, 232, 233, 234, 235, 238, 239, 240, 243, 248, 249, 251, 252, 254, 256, 257,
            258, 259, 261, 265, 269, 271, 272, 275, 277, 278, 279, 280, 281, 282, 284, 285, 286,
            287, 288, 290, 292, 295, 297, 299, 300, 301, 302, 303, 305, 306, 307, 308, 309, 310,
            312, 321, 322, 324, 331, 332, 334, 335, 336, 341, 342, 343, 344, 346, 347, 349, 352,
            354, 355, 356, 359, 360, 363, 365, 369, 372, 374, 376,
        ];
        // parameters from simulation
        let fragment_cost =
            TKF92FixedIndelCostBuilder::new(&[1.0, 1.1, 0.5], fragmentation, phylo_info)
                .build()
                .unwrap();
        // logl from simulation
        assert_relative_eq!(fragment_cost.logl(), -769.4115065236674, epsilon = 1e-10);
    }
}
