use approx::assert_relative_eq;
use assert_matches::assert_matches;
use nalgebra::DVector;

use crate::alignment::{Alignment, AncestralAlignment, Mapping, Sequences, MASA};
use crate::alphabets::Alphabet;
use crate::likelihood::{
    ModelSearchCost, PARAM_RANGE_POSITIVE, PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE,
};
use crate::optimisers::rooted_nni;
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{QMatrixMaker, SubstModel, SubstitutionCostBuilder as SCB};
use crate::substitution_models::{BLOSUM, GTR, HIVB, HKY, JC69, K80, TN93, WAG};
use crate::tkf_model::tkf92::TKF92IndelModel;
use crate::tkf_model::tkf92_fixed_fragmentation::TKF92FixedIndelModel;
use crate::tkf_model::tkf_indel::DUMMY_FREQS;
use crate::tree::NodeIdx::{self, Internal, Leaf};
use crate::{frequencies, record_wo_desc as record, tree, Error};

use super::*;

#[test]
fn tkf_dummy_freqs() {
    assert_eq!(&*DUMMY_FREQS, &DVector::<f64>::zeros(0));
}

#[cfg(test)]
pub(super) fn get_mapping_for_any_node<'a, AA: AncestralAlignment>(
    msa: &'a AA,
    node: &'a NodeIdx,
) -> &'a Mapping {
    match node {
        Internal(_) => msa.ancestral_map(node),
        Leaf(_) => msa.leaf_map(node),
    }
}

// This is a direct implementation of the TKF91 ln likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf91_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF91IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for i in 0..phylo.msa.len() {
        let mut event_prob = 1.0;
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[i].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu;
        }
        for node_idx in tree.postorder() {
            // skipping the root of the tree because it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let parent_is_gap = get_mapping_for_any_node(&phylo.msa, parent_id)[i].is_none();
            let current_is_gap = get_mapping_for_any_node(&phylo.msa, node_idx)[i].is_none();

            let beta = naive_beta(lambda, mu, time);
            if i == 0 {
                prob += ln_i1(lambda, beta.ln());
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= naive_h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= naive_n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += naive_ln_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= naive_n0(mu, beta).ln();
                }
                event_prob *= lambda * beta;
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
    }
    prob
}

// This is a direct implementation of the TKF92 ln likelihood calculation without any
// aggregation over subtrees and without substitutions. This direct calculation is sufficient
// if one is only interested in the indel likelihood for a fixed alignment and tree.
// Used for testing purposes only, i.e., to validate the aggregated implementation.
#[cfg(test)]
fn tkf92_indel_logl_without_aggregation<AA: AncestralAlignment>(
    model: &TKF92IndelModel,
    phylo: &PhyloInfo<AA>,
) -> f64 {
    let blocks = model.get_blocks(&phylo.msa);
    let tree = &phylo.tree;
    let lambda = model.lambda();
    let mu = model.mu();
    let r = model.params()[2];

    // for the root
    let mut prob: f64 = (1.0 - lambda / mu).ln();

    let mut last_event_deletion = vec![false; tree.len()];
    for (i, fragment) in blocks.iter().enumerate() {
        let mut event_prob = 1.0;
        let fragment_len = if i == 0 {
            *fragment
        } else {
            fragment - blocks[i - 1]
        };
        if get_mapping_for_any_node(&phylo.msa, &phylo.tree.root)[fragment - 1].is_some() {
            // the eq seq at the root has a fragment
            event_prob *= lambda / mu * (1.0 - r) / r;
            prob += fragment_len as f64 * r.ln();
        }
        for node_idx in tree.postorder() {
            // skipping the root of the tree because it has no parent and therefore also no
            // mutations probabilities
            if node_idx == &tree.root {
                continue;
            }
            let node_id_value = usize::from(node_idx);

            let time = tree.node(node_idx).blen;
            let parent_id = &tree.node(node_idx).parent.unwrap();
            let parent_is_gap =
                get_mapping_for_any_node(&phylo.msa, parent_id)[fragment - 1].is_none();
            let current_is_gap =
                get_mapping_for_any_node(&phylo.msa, node_idx)[fragment - 1].is_none();

            let beta = naive_beta(lambda, mu, time);
            if i == 0 {
                prob += ln_i1(lambda, beta.ln());
            }
            if parent_is_gap && current_is_gap {
                continue;
            } else if !parent_is_gap && !current_is_gap {
                // homolog block
                event_prob *= naive_h1(lambda, mu, beta, time);
                last_event_deletion[node_id_value] = false;
            } else if !parent_is_gap && current_is_gap {
                // deletion
                event_prob *= naive_n0(mu, beta);
                last_event_deletion[node_id_value] = true;
            } else if parent_is_gap && !current_is_gap {
                // insertion
                if last_event_deletion[node_id_value] {
                    prob += naive_ln_n1(lambda, mu, beta, time);
                    prob -= (lambda * beta).ln();
                    prob -= naive_n0(mu, beta).ln();
                }
                event_prob *= lambda * beta * (1.0 - r) / r;
                prob += fragment_len as f64 * r.ln();
                last_event_deletion[node_id_value] = false;
            }
        }
        prob += event_prob.ln();
        prob += (fragment_len - 1) as f64 * (1.0 + event_prob).ln();
    }
    prob
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_beta(lambda: f64, mu: f64, time: f64) -> f64 {
    let exp_term = ((lambda - mu) * time).exp();
    (1.0 - exp_term) / (mu - lambda * exp_term)
}

#[test]
fn tkf_beta_calculated_by_hand() {
    let correct = 0.5461782813185221;
    assert_relative_eq!(naive_beta(0.3, 0.5, 0.7), correct);
    assert_relative_eq!(ln_beta(0.3, 0.5, 0.7), correct.ln());
    assert_eq!(ln_beta(0.3, 0.5, 0.0), f64::NEG_INFINITY);
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_n0(mu: f64, beta: f64) -> f64 {
    mu * beta
}

#[test]
fn tkf_ln_n0_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = naive_beta(l, m, time);
    // (3(1-e^(-.5))/(3-2*e^(-.5)))
    let correct = 0.6605755607027574;
    assert_relative_eq!(naive_n0(m, b), correct);
    assert_relative_eq!(ln_n0(m, b.ln()), correct.ln());
    assert_eq!(ln_n0(m, f64::NEG_INFINITY), f64::NEG_INFINITY);
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_h1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    (-mu * time).exp() * (1.0 - lambda * beta)
}

#[test]
fn tkf_ln_h1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = naive_beta(l, m, time);
    // e^(-4.5) * (1-2(1-e^(-1.5))/(3-2*e^(-1.5)))
    let correct = 0.004350089645603061;
    assert_relative_eq!(naive_h1(l, m, b, time), correct);
    assert_relative_eq!(ln_h1(l, m, b.ln(), time), correct.ln());
    assert_eq!(ln_h1(l, m, f64::NEG_INFINITY, 0.0), 0.0);
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_ln_i1(lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta).ln()
}

#[test]
fn tkf_ln_i1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.0;
    let b = naive_beta(l, m, time);
    // ln((1-2(1-e^(-1))/(3-2*e^(-1)))
    let correct = -0.8172396554020775;
    assert_relative_eq!(naive_ln_i1(l, b), correct);
    assert_relative_eq!(ln_i1(l, b.ln()), correct);
    assert_eq!(ln_i1(l, f64::NEG_INFINITY), 0.0);
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_ln_n1(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let term1 = 1.0 - (-mu * time).exp() - mu * beta;
    let term2 = 1.0 - lambda * beta;
    (term1 * term2).ln()
}

#[test]
fn tkf_ln_n1_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 0.5;
    let b = naive_beta(l, m, time);
    // ln((1-e^(-1.5) - 3(1-e^(-.5))/(3-2*e^(-.5)) )* (1-2(1-e^(-.5))/(3-2*e^(-.5)))   (2(1-e^(-1))/(3-2*e^(-1)))^0)
    assert_relative_eq!(
        naive_ln_n1(l, m, b, time),
        -2.732135332549935,
        epsilon = 1e-14
    );
}

#[cfg(test)]
/// A direct implementation of the TKF function, not numerically stable, used for testing purposes only.
fn naive_eta(lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let mut e = naive_ln_n1(lambda, mu, beta, time);
    e -= lambda.ln() + beta.ln();
    e -= (mu * beta).ln();
    e
}

#[test]
fn tkf_eta_calculated_by_hand() {
    let l = 2.0;
    let m = 3.0;
    let time = 1.5;
    let b = naive_beta(l, m, time);
    // math.log( (1 - math.exp(-3*1.5) - 3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // * (1 - 2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5)))))
    // - math.log(2*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    // - math.log(3*((1 - math.exp((2-3)*1.5))/(3 - 2*math.exp((2-3)*1.5))))
    assert_relative_eq!(
        naive_eta(l, m, b, time),
        -2.922778333826742,
        epsilon = 1e-14
    );
    assert_relative_eq!(eta(l, m, b.ln(), time), -2.922778333826742, epsilon = 1e-14);
    assert_eq!(eta(l, m, f64::NEG_INFINITY, 0.0), -std::f64::consts::LN_2);
}

#[test]
fn tkf91_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A0", b"AAAB-D"),
        record!("B1", b"--ARAW"),
        record!("I1", b"AAAA-A"),
    ]);
    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF91IndelModel::default().get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, (1..msa.len() + 1).collect::<Vec<usize>>());
    assert_eq!(block_lens, vec![1; 6]);
}

#[test]
fn tkf92_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A0", b"AAB-D"),
        record!("B1", b"-ARAW"),
        record!("I1", b"AAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let blocks = TKF92IndelModel::default().get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, vec![1, 3, 4, 5]);
    assert_eq!(block_lens, vec![1, 2, 1, 1]);
}

#[test]
fn tkf92_fixed_get_blocks() {
    let tree = tree!("((A0:1.0,B1:1.0)I1:1.0);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A0", b"AAAAAAB-D"),
        record!("B1", b"---AAARAW"),
        record!("I1", b"AAAAAAA-A"),
    ]);

    let msa = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();

    let fragmentation = vec![1, 2, 7];
    let model = TKF92FixedIndelModel {
        params: vec![0.5, 1.5, 0.2],
        ln_r: 0.2_f64.ln(),
        fragmentation,
    };
    let blocks = model.get_blocks(&msa);
    let block_lens = get_block_lengths(&blocks);

    assert_eq!(blocks, vec![1, 2, 3, 7, 8, 9]);
    assert_eq!(block_lens, vec![1, 1, 1, 4, 1, 1]);
}

#[cfg(test)]
pub(super) fn setup_test_phylo(alphabet: &'static Alphabet) -> PhyloInfo<MASA> {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::with_alphabet_unchecked(
            vec![
                record!("A1", b"--GTGGA---"),
                record!("B2", b"-------NNA"),
                record!("I3", b"--T-------"),
                record!("C4", b"AGG-------"),
                record!("R5", b"--A-------"),
            ],
            alphabet,
        ),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn tkf_indel_get_and_set_params_and_freqs() {
    let mut tkf_indel_cost =
        TKF92IndelCostBuilder::new(&[1.0, 2.0, 0.3], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    // params
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.param(0), 1.0);
    assert_eq!(tkf_indel_cost.param(1), 2.0);
    assert_eq!(tkf_indel_cost.param(2), 0.3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.3);
    tkf_indel_cost.set_param(2, 0.33);
    assert_eq!(tkf_indel_cost.param_count(), 3);
    assert_eq!(tkf_indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_indel_cost.model.r(), 0.33);
    // freqs
    assert_eq!(tkf_indel_cost.freqs(), &*DUMMY_FREQS);
    assert_eq!(
        tkf_indel_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf_get_and_set_params() {
    let subst_model =
        SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]).unwrap();
    let mut tkf_cost = TKF92CostBuilder::new(
        &[1.0, 2.0, 0.3],
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.3);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.7);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);
    assert_eq!(tkf_cost.indel_cost.model.lambda(), 1.0);
    assert_eq!(tkf_cost.indel_cost.model.mu(), 2.0);
    assert_eq!(tkf_cost.indel_cost.model.r(), 0.3);
    tkf_cost.set_param(2, 0.33);
    tkf_cost.set_param(5, 0.77);
    assert_eq!(tkf_cost.param_count(), 8);
    assert_eq!(tkf_cost.param(0), 1.0);
    assert_eq!(tkf_cost.param(1), 2.0);
    assert_eq!(tkf_cost.param(2), 0.33);
    assert_eq!(tkf_cost.param(3), 0.5);
    assert_eq!(tkf_cost.param(4), 0.6);
    assert_eq!(tkf_cost.param(5), 0.77);
    assert_eq!(tkf_cost.param(6), 0.8);
    assert_eq!(tkf_cost.param(7), 0.9);

    assert_eq!(
        tkf_cost.empirical_freqs(),
        setup_test_phylo(Alphabet::dna()).freqs()
    );
}

#[test]
fn tkf91_indel_cost_fmt() {
    let tkf_indel_cost = TKF91IndelCostBuilder::new(&[1.0, 2.0], setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2");
}

#[test]
fn tkf91_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let tkf_cost =
        TKF91CostBuilder::new(&[1.0, 2.0], subst_model, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF91 with lambda = 1, mu = 2 and JC69");
}

#[test]
fn tkf92_indel_cost_fmt() {
    let tkf_indel_cost =
        TKF92IndelCostBuilder::new(&[1.0, 2.0, 0.3], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();

    let fmt = format!("{}", tkf_indel_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3");
}

#[test]
fn tkf92_cost_fmt() {
    let subst_model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let tkf_cost = TKF92CostBuilder::new(
        &[1.0, 2.0, 0.3],
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();

    let fmt = format!("{}", tkf_cost);

    assert_eq!(fmt, "TKF92 with lambda = 1, mu = 2, r = 0.3 and JC69");
}

#[test]
fn tkf_get_and_set_freqs() {
    let subst_model =
        SubstModel::<GTR>::new(&[0.1, 0.2, 0.3, 0.4], &[0.5, 0.6, 0.7, 0.8, 0.9]).unwrap();
    let mut tkf_cost = TKF92CostBuilder::new(
        &[1.0, 2.0, 0.3],
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.1, 0.2, 0.3, 0.4]);
    tkf_cost
        .set_freqs(frequencies!(&[0.4, 0.3, 0.2, 0.1]))
        .unwrap();
    assert_eq!(tkf_cost.freqs().as_slice(), &[0.4, 0.3, 0.2, 0.1]);
}

#[test]
fn tkf91_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();
    let tkf_cost =
        TKF91CostBuilder::new(&[1.0, 2.0], subst_model, setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    let lambda_range = tkf_cost.param_range(usize::from(TKF91Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = tkf_cost.param_range(usize::from(TKF91Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);

    for subst_param_idx in 2..tkf_cost.param_count() {
        let subst_range = tkf_cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[cfg(test)]
fn tkf92_subst_param_range<Q: QMatrix, T: TKFModel, AA: AncestralAlignment>(
    cost: &TKFCost<Q, T, AA>,
) {
    for subst_param_idx in 3..cost.param_count() {
        let subst_range = cost.param_range(subst_param_idx);
        let true_subst_range = PARAM_RANGE_POSITIVE;
        assert_eq!(subst_range, true_subst_range);
    }
}

#[cfg(test)]
fn tkf92_indel_param_range<T: TKFModel, AA: AncestralAlignment>(cost: &TKFIndelCost<T, AA>) {
    let lambda_range = cost.param_range(usize::from(TKF92Parameters::Lambda));
    let true_lambda_range = (f64::EPSILON, 2.0 - f64::EPSILON);
    assert_eq!(lambda_range, true_lambda_range);
    let mu_range = cost.param_range(usize::from(TKF92Parameters::Mu));
    let true_mu_range = (1.0 + f64::EPSILON, f64::MAX);
    assert_eq!(mu_range, true_mu_range);
    let r_range = cost.param_range(usize::from(TKF92Parameters::R));
    let true_r_range = PARAM_RANGE_UNIT_INTERVAL_EXCLUSIVE;
    assert_eq!(r_range, true_r_range);
}
#[test]
fn tkf92_param_range() {
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();
    let tkf_cost = TKF92CostBuilder::new(
        &[1.0, 2.0, 0.3],
        subst_model,
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    tkf92_subst_param_range(&tkf_cost);
    tkf92_indel_param_range(&tkf_cost.indel_cost);
}

#[test]
fn tkf92_fixed_param_range() {
    let tkf_cost = TKF92FixedIndelCostBuilder::new(
        &[1.0, 2.0, 0.3],
        vec![],
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    tkf92_indel_param_range(&tkf_cost);
}

#[test]
fn tkf92_add_param_range() {
    let tkf_cost = TKF92IndelAddBlocksCostBuilder::new(
        &[1.0, 2.0, 0.3],
        vec![],
        setup_test_phylo(Alphabet::dna()),
    )
    .build()
    .unwrap();
    tkf92_indel_param_range(&tkf_cost);
}

#[test]
fn tkf91_indel_logl_manual() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let tree = phylo.tree.clone();
    let lambda = 0.1;
    let mu = 0.2;
    let tkf91_cost = TKF91IndelCostBuilder::new(&[lambda, mu], phylo)
        .build()
        .unwrap();

    let logl = tkf91_cost.logl();

    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    // immortal links
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 2.0;
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu;
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= naive_n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen);
    manual_calculation += x.ln() * 4.0;
    // fourth block ([7:10], insertion at B2)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("B2").blen);
    manual_calculation += x.ln() * 3.0;
    manual_calculation += naive_ln_n1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= naive_n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();

    assert_relative_eq!(logl, manual_calculation, epsilon = 1e-12);
}

#[test]
fn tkf91_indel_logl_half_manual() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let lambda = 0.1;
    let mu = 0.2;
    let tkf91_cost = TKF91IndelCostBuilder::new(&[lambda, mu], phylo)
        .build()
        .unwrap();

    let logl = tkf91_cost.logl();
    let half_manual = tkf91_indel_logl_without_aggregation(&tkf91_cost.model, &tkf91_cost.phylo);

    assert_relative_eq!(logl, half_manual, epsilon = 1e-11);
}

#[test]
fn tkf92_indel_logl_manual() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let tree = phylo.tree.clone();
    let m = phylo.msa.len() as f64;
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo)
        .build()
        .unwrap();

    let logl = tkf92_cost.logl();

    let mut manual_calculation = 0.0;
    manual_calculation += (1.0 - lambda / mu).ln();
    manual_calculation += m * r.ln();
    // immortal links
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("A1").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("I3").blen));
    manual_calculation += ln_i1(lambda, ln_beta(lambda, mu, tree.by_id("C4").blen));
    // first block ([0:2], insertion at C4)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 1.0 * (1.0 + x).ln();
    // second block ([2:3], all homologous except B2 deleted)
    let mut x = lambda / mu * (1.0 - r) / r;
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("C4").blen),
        tree.by_id("C4").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("A1").blen),
        tree.by_id("A1").blen,
    );
    x *= naive_h1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("I3").blen),
        tree.by_id("I3").blen,
    );
    x *= naive_n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen));
    manual_calculation += x.ln();
    // third block ([3:7], insertion at A1)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("C4").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 3.0 * (1.0 + x).ln();
    // fourth block ([7:10], insertion at B2)
    let x = lambda * naive_beta(lambda, mu, tree.by_id("B2").blen) * (1.0 - r) / r;
    manual_calculation += x.ln() + 2.0 * (1.0 + x).ln();
    manual_calculation += naive_ln_n1(
        lambda,
        mu,
        naive_beta(lambda, mu, tree.by_id("B2").blen),
        tree.by_id("B2").blen,
    );
    manual_calculation -= naive_n0(mu, naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();
    manual_calculation -= (lambda * naive_beta(lambda, mu, tree.by_id("B2").blen)).ln();

    assert_relative_eq!(logl, manual_calculation, epsilon = 1e-12);
}

#[test]
fn tkf92_indel_logl_half_manual() {
    let phylo = setup_test_phylo(Alphabet::dna());
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo)
        .build()
        .unwrap();

    let logl = tkf92_cost.logl();
    let half_manual = tkf92_indel_logl_without_aggregation(&tkf92_cost.model, &tkf92_cost.phylo);

    assert_relative_eq!(logl, half_manual, epsilon = 1e-11);
}

#[test]
fn tkf91_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();

    let tkf91_err = TKF91CostBuilder::new(&[0.1, 0.2], subst_model, phylo).build();

    assert_matches!(
        tkf91_err, Err(Error::Alphabet(msg)) if msg.contains(
        "alphabet mismatch between model and alignment")
    );
}

#[test]
fn tkf91_build_default() {
    let tkf_indel_cost = TKF91IndelCostBuilder::new(&[], setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    assert_eq!(tkf_indel_cost.model.lambda(), DEFAULT_LAMBDA);
    assert_eq!(tkf_indel_cost.model.mu(), DEFAULT_MU);
}

#[test]
fn tkf91_build_default_one_param() {
    let tkf_indel_cost = TKF91IndelCostBuilder::new(&[0.0331], setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    assert_eq!(tkf_indel_cost.model.lambda(), DEFAULT_LAMBDA);
    assert_eq!(tkf_indel_cost.model.mu(), DEFAULT_MU);
}

#[test]
fn tkf92_cost_builder_fails() {
    let phylo = setup_test_phylo(Alphabet::protein());
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();

    let tkf92_err = TKF92CostBuilder::new(&[0.1, 0.2, 0.3], subst_model, phylo).build();

    assert_matches!(
        tkf92_err, Err(Error::Alphabet(msg)) if msg.contains(
        "alphabet mismatch between model and alignment")
    );
}

#[test]
fn tkf92_build_default() {
    let tkf_indel_cost = TKF92IndelCostBuilder::new(&[], setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    assert_eq!(tkf_indel_cost.model.lambda(), DEFAULT_LAMBDA);
    assert_eq!(tkf_indel_cost.model.mu(), DEFAULT_MU);
    assert_eq!(tkf_indel_cost.model.r(), DEFAULT_R);
}

#[test]
fn tkf92_build_default_one_param() {
    let tkf_indel_cost = TKF92IndelCostBuilder::new(&[0.0331], setup_test_phylo(Alphabet::dna()))
        .build()
        .unwrap();
    assert_eq!(tkf_indel_cost.model.lambda(), DEFAULT_LAMBDA);
    assert_eq!(tkf_indel_cost.model.mu(), DEFAULT_MU);
    assert_eq!(tkf_indel_cost.model.r(), DEFAULT_R);
}

#[test]
fn tkf92_fixed_build_default() {
    let tkf_indel_cost =
        TKF92FixedIndelCostBuilder::new(&[], vec![], setup_test_phylo(Alphabet::dna()))
            .build()
            .unwrap();
    assert_eq!(tkf_indel_cost.model.lambda(), DEFAULT_LAMBDA);
    assert_eq!(tkf_indel_cost.model.mu(), DEFAULT_MU);
    assert_eq!(tkf_indel_cost.model.r(), DEFAULT_R);
}

#[test]
fn tkf91_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
    let subst_model =
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]).unwrap();
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let tkf_cost = TKF91CostBuilder::new(&[lambda, mu], subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = ModelSearchCost::cost(&tkf_cost);
    let subst_logl = ModelSearchCost::cost(&subst_cost);
    let tkf_logl = tkf91_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl);
}

#[test]
fn tkf92_logl_with_substitution() {
    // arrange
    let phylo = setup_test_phylo(Alphabet::dna());
    let subst_model =
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]).unwrap();
    let subst_cost = SCB::new(subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_cost = TKF92CostBuilder::new(&[lambda, mu, r], subst_model, phylo)
        .build()
        .unwrap();

    // act
    let logl = ModelSearchCost::cost(&tkf_cost);
    let subst_logl = ModelSearchCost::cost(&subst_cost);
    let tkf_logl = tkf92_indel_logl_without_aggregation(
        &tkf_cost.indel_cost.model,
        &tkf_cost.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(logl, subst_logl + tkf_logl, epsilon = 1e-12);
}

#[test]
fn tkf_indel_history_doesnt_change_felsenstein() {
    // arrange
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--N-------"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--N-------"),
    ]);
    let seqs2 = Sequences::new_unchecked(vec![
        record!("A1", b"--GTGTA---"),
        record!("B2", b"-------AGT"),
        record!("I3", b"--NNNNNNNN"),
        record!("C4", b"GTA-------"),
        record!("R5", b"--NNNNN---"),
    ]);
    let msa1 = MASA::from_aligned_with_ancestral(seqs, &tree).unwrap();
    let msa2 = MASA::from_aligned_with_ancestral(seqs2, &tree).unwrap();
    let phylo1 = PhyloInfo {
        msa: msa1,
        tree: tree.clone(),
    };
    let phylo2 = PhyloInfo { msa: msa2, tree };
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let subst_model =
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0]).unwrap();
    let tkf_cost1 = TKF92CostBuilder::new(&[lambda, mu, r], subst_model.clone(), phylo1)
        .build()
        .unwrap();

    let tkf_cost2 = TKF92CostBuilder::new(&[lambda, mu, r], subst_model, phylo2)
        .build()
        .unwrap();

    // act
    let tkf_log_1 = ModelSearchCost::cost(&tkf_cost1.clone());
    let tkf_log_2 = ModelSearchCost::cost(&tkf_cost2.clone());
    let tkf_indel_cost_1 = ModelSearchCost::cost(&tkf_cost1.indel_cost);
    let tkf_indel_cost_without_agg_1 = tkf92_indel_logl_without_aggregation(
        &tkf_cost1.indel_cost.model,
        &tkf_cost1.indel_cost.phylo,
    );
    let tkf_indel_cost_2 = ModelSearchCost::cost(&tkf_cost2.indel_cost);
    let tkf_indel_cost_without_agg_2 = tkf92_indel_logl_without_aggregation(
        &tkf_cost2.indel_cost.model,
        &tkf_cost2.indel_cost.phylo,
    );

    // assert
    assert_relative_eq!(tkf_indel_cost_1, tkf_indel_cost_without_agg_1);
    assert_relative_eq!(tkf_indel_cost_2, tkf_indel_cost_without_agg_2);
    assert_relative_eq!(tkf_log_1 - tkf_indel_cost_1, tkf_log_2 - tkf_indel_cost_2);
}

#[cfg(test)]
fn modify_tkf92_subst_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_original_param = 1.0;
    let subst_changed_param = 0.5;
    let subst_model = SubstModel::<Q>::new(&[], &[subst_original_param]).unwrap();
    let mut tkf_cost = TKF92CostBuilder::new(&[0.1, 0.2, 0.3], subst_model, phylo.clone())
        .build()
        .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(3, subst_changed_param);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[subst_changed_param]).unwrap();
    let tkf_cost = TKF92CostBuilder::new(&[0.1, 0.2, 0.3], subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[cfg(test)]
fn modify_tkf92_indel_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let phylo = setup_test_phylo(Q::alphabet());
    let subst_model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let tkf_original_mu = 0.2;
    let tkf_changed_mu = 0.25;
    let mut tkf_cost =
        TKF92CostBuilder::new(&[0.1, tkf_original_mu, 0.3], subst_model, phylo.clone())
            .build()
            .unwrap();

    // sanity check
    let logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl, ModelSearchCost::cost(&tkf_cost));

    // The likelihood should change if we change model parameters
    tkf_cost.set_param(1, tkf_changed_mu);
    let logl2 = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(logl2, ModelSearchCost::cost(&tkf_cost));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let subst_model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let tkf_cost = TKF92CostBuilder::new(&[0.1, tkf_changed_mu, 0.3], subst_model, phylo)
        .build()
        .unwrap();
    let new_logl = ModelSearchCost::cost(&tkf_cost);
    assert_eq!(new_logl, ModelSearchCost::cost(&tkf_cost));
    assert_eq!(logl2, new_logl);
}

#[test]
fn tkf92_modify_subst_model_params_costs_match() {
    modify_tkf92_subst_params_costs_match_template::<K80>();
    modify_tkf92_subst_params_costs_match_template::<HKY>();
    modify_tkf92_subst_params_costs_match_template::<TN93>();
    modify_tkf92_subst_params_costs_match_template::<GTR>();
}

#[test]
fn tkf_modify_indel_model_params_costs_match() {
    modify_tkf92_indel_params_costs_match_template::<JC69>();
    modify_tkf92_indel_params_costs_match_template::<K80>();
    modify_tkf92_indel_params_costs_match_template::<HKY>();
    modify_tkf92_indel_params_costs_match_template::<TN93>();
    modify_tkf92_indel_params_costs_match_template::<GTR>();
    modify_tkf92_indel_params_costs_match_template::<WAG>();
    modify_tkf92_indel_params_costs_match_template::<BLOSUM>();
    modify_tkf92_indel_params_costs_match_template::<HIVB>();
}

#[test]
fn tkf_update_tree() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"--GTGGATGC"),
            record!("B2", b"--G----CGA"),
            record!("I3", b"--N----NNN"),
            record!("C4", b"AGC-------"),
            record!("R5", b"--N-------"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let mut tkf_cost = TKF92CostBuilder::new(&[lambda, mu, r], subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let original_logl = TreeSearchCost::cost(&tkf_cost);
    assert_ne!(original_logl, f64::NEG_INFINITY);

    let node_idx = &phylo.tree.by_id("I3").idx;
    let child_idx = &phylo.tree.by_id("A1").idx;
    let new_tree = rooted_nni(&phylo.tree, node_idx, child_idx).unwrap();
    let new_tree_newick = new_tree.to_newick();
    tkf_cost.update_tree(new_tree);

    assert_eq!(new_tree_newick, tkf_cost.tree().to_newick());
    let new_logl = TreeSearchCost::cost(&tkf_cost);

    let new_phylo = PhyloInfo {
        msa: tkf_cost.masa().clone(),
        tree: tkf_cost.tree().clone(),
    };
    let clean_cost = TKF92CostBuilder::new(&[lambda, mu, r], subst_model.clone(), new_phylo)
        .build()
        .unwrap();
    let clean_logl = TreeSearchCost::cost(&clean_cost);
    assert_ne!(original_logl, new_logl);
    assert_eq!(new_logl, clean_logl);
}

#[cfg(test)]
fn setup_short_branches_phylo() -> PhyloInfo<MASA> {
    let tree = tree!("(((A1:1e-20,B2:2.0)I3:1e-16,C4:2.0)R5:0.0);");
    let msa = MASA::from_aligned_with_ancestral(
        // Testing all events on short branches
        Sequences::new_unchecked(vec![
            record!("A1", b"-AA-AA"),
            record!("B2", b"A-A--A"),
            record!("I3", b"AAAA-A"),
            record!("C4", b"-----A"),
            record!("R5", b"-----A"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn tkf92_underflow_short_branches() {
    let phylo = setup_short_branches_phylo();
    let lambda = 1.0;
    let mu = 4.1;
    let r = 0.8;
    let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo)
        .build()
        .unwrap();
    let logl = tkf92_cost.logl();
    assert!(!logl.is_nan());
    assert!(logl.is_finite());
}

#[test]
fn tkf92_underflow_short_branches_large_mu() {
    let phylo = setup_short_branches_phylo();
    let lambda = 1.0;
    let mu = 10000.0;
    let r = 0.8;
    let tkf92_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo)
        .build()
        .unwrap();
    let logl = tkf92_cost.logl();
    assert!(!logl.is_nan());
    assert!(logl.is_finite());
}
