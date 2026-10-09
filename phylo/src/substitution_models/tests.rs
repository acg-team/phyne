use std::ops::Mul;
use std::path::Path;

use approx::assert_relative_eq;
use assert_matches::assert_matches;

use crate::alignment::{Alignment, Sequences, MSA};
use crate::alphabets::{Alphabet, AMINOACIDS, GAP};
use crate::evolutionary_models::EvoModel;
use crate::io::read_sequences;
use crate::likelihood::ModelSearchCost;
use crate::parsimony::{DiagonalZeros as Z, ParsimonyModel, Rounding as R};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::substitution_models::{
    dna_models::{GTR, HKY, JC69, K80, TN93},
    protein_models::{BLOSUM, HIVB, WAG},
    FreqVector, ProbabilityMatrix, QMatrix, QMatrixMaker, RateMatrix, SubstModel,
    SubstitutionCostBuilder as SCB,
};
use crate::{record_wo_desc as record, tree, Error, SubstitutionModelError};

use super::{
    DEFAULT_GTR_RATES, DEFAULT_TN93_RATES, DEFAULT_TS_TV_RATIO, GTR_PARAM_N, HKY_PARAM_N,
    K80_PARAM_N, TN93_PARAM_N,
};

#[cfg(test)]
const EQUAL_DNA_FREQS: [f64; 4] = [0.25; 4];

#[cfg(test)]
const EQUAL_PROTEIN_FREQS: [f64; 20] = [1.0 / 20.0; 20];

#[cfg(test)]
fn freqs_cannot_change_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // freqs should not change for JC69 and K80
    assert!(model
        .set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]))
        .is_err());
    assert_eq!(model.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
}

#[test]
fn dna_freqs_cannot_change() {
    freqs_cannot_change_template(SubstModel::<JC69>::default());
    freqs_cannot_change_template(SubstModel::<K80>::default());
}

#[cfg(test)]
fn freqs_fixed_equal_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // freqs can be set to equal values
    assert!(model.set_freqs(frequencies!(&EQUAL_DNA_FREQS)).is_ok());
    assert_eq!(model.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
}

#[test]
fn dna_freqs_fixed_equal() {
    freqs_fixed_equal_template(SubstModel::<JC69>::default());
    freqs_fixed_equal_template(SubstModel::<K80>::default());
}

#[cfg(test)]
fn freqs_updated_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // freqs should change for HKY, TN93, and GTR
    let new_freqs = frequencies!(&[0.1, 0.2, 0.3, 0.4]);
    assert!(model.set_freqs(new_freqs.clone()).is_ok());
    assert_eq!(model.freqs(), &new_freqs);
    assert_ne!(model.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
}

#[test]
fn dna_freqs_updated() {
    freqs_updated_template(SubstModel::<HKY>::default());
    freqs_updated_template(SubstModel::<TN93>::default());
    freqs_updated_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn freqs_not_updated_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // freqs should not change when set to invalid values
    let correct_freqs = model.freqs().clone();
    assert!(model
        .set_freqs(frequencies!(&[-1.0, 1.5, -0.3, 0.8]))
        .is_err());

    assert_eq!(model.freqs(), &correct_freqs);
}

#[test]
fn dna_freqs_not_updated() {
    freqs_not_updated_template(SubstModel::<JC69>::default());
    freqs_not_updated_template(SubstModel::<K80>::default());
    freqs_not_updated_template(SubstModel::<HKY>::default());
    freqs_not_updated_template(SubstModel::<TN93>::default());
    freqs_not_updated_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn protein_freqs_not_updated_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // freqs should not change when set to invalid values
    let correct_freqs = model.freqs().clone();
    assert!(model.set_freqs(frequencies!(&[-0.1; 20])).is_err());

    assert_eq!(model.freqs(), &correct_freqs);
}

#[test]
fn protein_freqs_not_updated() {
    protein_freqs_not_updated_template(SubstModel::<WAG>::default());
    protein_freqs_not_updated_template(SubstModel::<HIVB>::default());
    protein_freqs_not_updated_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn param_fixed_template<Q: QMatrix>(mut model: SubstModel<Q>) {
    // parameters should not change for JC69
    model.set_param(0, 66.0);
    assert!(model.params().is_empty());
}

#[test]
fn dna_params_fixed() {
    param_fixed_template(SubstModel::<JC69>::default());
}

#[cfg(test)]
fn params_updated_template<Q: QMatrix>(mut model: SubstModel<Q>, new_params: &[f64]) {
    // parameters should change for K80, HKY, TN93, and GTR
    for (i, &param) in new_params.iter().enumerate() {
        model.set_param(i, param);
    }
    assert_eq!(model.params(), new_params);
}

#[test]
fn dna_params_updated() {
    params_updated_template(SubstModel::<HKY>::default(), &[5.0]);
    params_updated_template(SubstModel::<K80>::default(), &[5.0]);
    params_updated_template(SubstModel::<TN93>::default(), &[2.0, 0.1]);
    params_updated_template(SubstModel::<GTR>::default(), &[0.9; 5]);
}

#[test]
fn dna_jc69_correct() {
    let jc69 = SubstModel::<JC69>::default();
    let jc69_2 = SubstModel::<JC69>::new(&EQUAL_DNA_FREQS, &[]).unwrap();
    assert_eq!(jc69, jc69_2);
    assert_relative_eq!(jc69.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(jc69.rate(b'A', b'C'), 1.0 / 3.0);
    assert_relative_eq!(jc69.rate(b'G', b'T'), 1.0 / 3.0);
    assert_relative_eq!(jc69.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
}

#[test]
fn dna_jc69_params() {
    let jc69 = SubstModel::<JC69>::default();
    assert_relative_eq!(jc69.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_eq!(format!("{jc69}"), "JC69".to_string());
}

#[test]
fn dna_jc69_invalid_inputs() {
    assert_matches!(
        SubstModel::<JC69>::new(&[0.1, 0.2, 0.3, 0.4], &[]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::UnequalFrequencies { .. }
        ))
    );

    assert_matches!(
        SubstModel::<JC69>::new(&EQUAL_DNA_FREQS, &[0.5]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::ParameterCount { .. }
        ))
    );
}

#[test]
fn dna_k80_default() {
    let k80 = SubstModel::<K80>::default();
    let k801 = SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[DEFAULT_TS_TV_RATIO]).unwrap();
    assert_eq!(k80, k801);
    assert_relative_eq!(k80.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(k80.rate(b'T', b'A'), 1.0 * 0.25);
    assert_relative_eq!(k80.rate(b'A', b'G'), 2.0 * 0.25);
    assert_relative_eq!(k80.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_eq!(k80.params(), &[DEFAULT_TS_TV_RATIO]);
}

#[test]
fn dna_k80_custom() {
    let k80 = SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[0.75]).unwrap();
    assert_relative_eq!(k80.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_relative_eq!(k80.params(), &[0.75].as_slice());
    assert_eq!(format!("{k80}"), format!("K80 with [kappa = {:.5}]", 0.75));
}

#[test]
fn dna_k80_invalid_inputs() {
    assert_matches!(
        SubstModel::<K80>::new(&[0.1, 0.2, 0.3, 0.4], &[]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::UnequalFrequencies { .. }
        ))
    );

    assert_matches!(
        SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[0.5, 0.3]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::ParameterCount { .. }
        ))
    );
}

#[cfg(test)]
fn check_freq_convergence(p: ProbabilityMatrix, pi: &FreqVector, epsilon: f64) {
    assert_eq!(p.nrows(), pi.len());
    assert_eq!(p.ncols(), pi.len());
    for row in p.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(row, pi.transpose().as_view(), epsilon = epsilon);
    }
}

#[cfg(test)]
fn infinity_p_template<Q: QMatrix>(model: SubstModel<Q>) {
    let p_inf = model.p(1000.0);
    assert_eq!(p_inf.shape(), model.q().shape());
    check_freq_convergence(p_inf, model.freqs(), 1e-5);
}

#[test]
fn dna_infinity_p() {
    infinity_p_template(SubstModel::<JC69>::new(&EQUAL_DNA_FREQS, &[]).unwrap());
    infinity_p_template(SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[0.5]).unwrap());
    infinity_p_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap());
    infinity_p_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    );
    infinity_p_template(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[test]
fn wag_infinity_p_convergence() {
    infinity_p_template(SubstModel::<WAG>::new(&EQUAL_PROTEIN_FREQS, &[]).unwrap());
    infinity_p_template(SubstModel::<WAG>::default());
}

#[test]
fn hivb_infinity_p_convergence() {
    infinity_p_template(SubstModel::<HIVB>::new(&EQUAL_PROTEIN_FREQS, &[]).unwrap());
    infinity_p_template(SubstModel::<HIVB>::default());
}

#[test]
fn blosum_infinity_p_convergence() {
    infinity_p_template(SubstModel::<BLOSUM>::new(&EQUAL_PROTEIN_FREQS, &[]).unwrap());
    infinity_p_template(SubstModel::<BLOSUM>::default());
}

#[test]
fn dna_hky_default() {
    let hky = SubstModel::<HKY>::default();
    let hky2 = SubstModel::<HKY>::new(&EQUAL_DNA_FREQS, &[DEFAULT_TS_TV_RATIO]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_eq!(hky.params(), &[DEFAULT_TS_TV_RATIO]);
    assert_eq!(hky, hky2);
}

#[test]
fn dna_hky_custom() {
    let hky = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap();
    let hky2 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap();
    assert_relative_eq!(hky.freqs(), &frequencies!(&[0.22, 0.26, 0.33, 0.19]));
    assert_eq!(hky.params(), &[0.5]);
    assert_relative_eq!(
        hky.q()
            .diagonal()
            .component_mul(&frequencies!(&[0.22, 0.26, 0.33, 0.19]))
            .sum(),
        -1.0
    );
    assert_eq!(hky, hky2);
}

#[test]
fn dna_hky_invalid_inputs() {
    assert_matches!(
        SubstModel::<HKY>::new(&[0.1, 0.2, 0.3, 0.4], &[2.0, 8.0]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::ParameterCount { .. }
        ))
    );
}

#[test]
fn dna_gtr_default() {
    let gtr = SubstModel::<GTR>::default();
    let gtr2 = SubstModel::<GTR>::new(&EQUAL_DNA_FREQS, &DEFAULT_GTR_RATES).unwrap();
    assert_eq!(gtr.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_eq!(gtr.params(), &DEFAULT_GTR_RATES);
    assert_eq!(gtr.q()[(0, 0)], -1.0);
    assert!(gtr.rate(b'T', b'T') < 0.0);
    assert!(gtr.rate(b'A', b'A') < 0.0);
    assert_relative_eq!(gtr.rate(b'T', b'C'), gtr.rate(b'C', b'T'));
    assert_relative_eq!(gtr.rate(b'A', b'G'), gtr.rate(b'G', b'A'));
    assert_eq!(gtr, gtr2);
}

#[test]
fn dna_gtr_invalid_inputs() {
    assert_matches!(
        SubstModel::<GTR>::new(&[], &[2.0, 1.0, 3.0, 6.0, 0.5]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::FrequencyCount { .. }
        ))
    );

    assert_matches!(
        SubstModel::<GTR>::new(&[-0.1, 0.26, 0.33, 0.19], &[2.0, 1.0, 3.0, 6.0, 0.5]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::NegativeFrequency { .. }
        ))
    );

    assert_matches!(
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 1.19], &[2.0, 1.0, 3.0, 6.0, 0.5]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::FrequencySum { .. }
        ))
    );

    assert_matches!(
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5; 3]),
        Err(Error::SubstitutionModel(
            SubstitutionModelError::ParameterCount { .. }
        ))
    );
}

#[test]
fn dna_tn93_correct() {
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[442.29, 217.81]).unwrap();
    let expected_pi = frequencies!(&[0.22, 0.26, 0.33, 0.19]);
    let expected_q = RateMatrix::from_column_slice(
        4,
        4,
        &[
            -1.4732124694954951,
            1.2409529074850258,
            0.002805744890196536,
            0.002805744890196536,
            1.4665807088459397,
            -1.2475846681345815,
            0.0033158803247777245,
            0.0033158803247777245,
            0.0042086173352948045,
            0.0042086173352948045,
            -0.5339064704940854,
            0.9166789418005613,
            0.002423143314260645,
            0.002423143314260645,
            0.5277848452791111,
            -0.9228005670155355,
        ],
    );
    assert_relative_eq!(tn93.q(), &expected_q);
    assert_relative_eq!(tn93.q().diagonal().component_mul(&expected_pi).sum(), -1.0);
    assert_relative_eq!(tn93.freqs(), &expected_pi);
    assert_relative_eq!(
        tn93.rate(b'T', b'C') * expected_pi[0],
        tn93.rate(b'C', b'T') * expected_pi[1],
    );
    assert_relative_eq!(
        tn93.rate(b'A', b'G') * expected_pi[2],
        tn93.rate(b'G', b'A') * expected_pi[3],
    );
    assert_relative_eq!(
        tn93.rate(b'T', b'A') * expected_pi[0],
        tn93.rate(b'A', b'T') * expected_pi[2],
    );
    assert_relative_eq!(
        tn93.rate(b'C', b'G') * expected_pi[1],
        tn93.rate(b'G', b'C') * expected_pi[3],
    );
}

#[test]
fn dna_tn93_default() {
    let tn93 = SubstModel::<TN93>::default();
    let tn932 = SubstModel::<TN93>::new(&EQUAL_DNA_FREQS, &DEFAULT_TN93_RATES).unwrap();
    assert_eq!(tn93.freqs(), &frequencies!(&EQUAL_DNA_FREQS));
    assert_eq!(tn93.params(), &DEFAULT_TN93_RATES);
    assert_eq!(tn93, tn932);
}

#[test]
fn dna_tn93_incorrect() {
    assert!(SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915]).is_err());

    assert!(SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 1.19], &[0.5, 0.6, 0.3]).is_err());

    assert!(SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.6, 0.3, 0.56]).is_err());

    assert!(SubstModel::<TN93>::new(&[], &[2.0, 1.0, 3.0]).is_err());
}

#[test]
fn dna_normalisation() {
    let jc69 = SubstModel::<JC69>::new(&EQUAL_DNA_FREQS, &[]).unwrap();
    assert_relative_eq!(jc69.q().diagonal().component_mul(jc69.freqs()).sum(), -1.0);
    let k80 = SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[3.0]).unwrap();
    assert_relative_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    let hky = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7]).unwrap();
    assert_relative_eq!(hky.q().diagonal().component_mul(hky.freqs()).sum(), -1.0);
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.59, 0.29]).unwrap();
    assert_relative_eq!(tn93.q().diagonal().component_mul(tn93.freqs()).sum(), -1.0);
    let gtr = SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7; 5]).unwrap();
    assert_relative_eq!(gtr.q().diagonal().component_mul(gtr.freqs()).sum(), -1.0);
}

#[test]
fn dna_normalised_param_change() {
    let mut k80 = SubstModel::<K80>::new(&EQUAL_DNA_FREQS, &[3.0]).unwrap();
    assert_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    let k80_old = k80.clone();
    assert_eq!(k80.params(), &[3.0]);

    k80.set_param(0, 10.0);
    assert_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    assert_eq!(k80.params(), &[10.0]);
    assert_ne!(k80.params(), k80_old.params());
    assert_ne!(k80.q(), k80_old.q());
}

#[cfg(test)]
fn protein_correct_access_template<Q: QMatrix>(model: SubstModel<Q>) {
    // Access all rates to ensure correct indexing
    for i in AMINOACIDS.iter() {
        for j in AMINOACIDS.iter() {
            model.rate(*i, *j);
        }
    }
}

#[test]
fn protein_correct_access() {
    protein_correct_access_template(SubstModel::<WAG>::default());
    protein_correct_access_template(SubstModel::<HIVB>::default());
    protein_correct_access_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn protein_gap_access_template<Q: QMatrix>(model: SubstModel<Q>) {
    model.rate(GAP, b'L');
}

#[test]
#[should_panic]
fn protein_incorrect_access_wag() {
    protein_gap_access_template(SubstModel::<WAG>::default());
}

#[test]
#[should_panic]
fn protein_incorrect_access_hivb() {
    protein_gap_access_template(SubstModel::<HIVB>::default());
}

#[test]
#[should_panic]
fn protein_incorrect_access_blosum() {
    protein_gap_access_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn normalised_template<Q: QMatrix>(model: SubstModel<Q>) {
    assert_relative_eq!(
        (model.q().diagonal().transpose().mul(model.freqs()))[(0, 0)],
        -1.0,
        epsilon = 1e-10
    );
}

#[test]
fn protein_normalised() {
    normalised_template(SubstModel::<WAG>::default());
    normalised_template(SubstModel::<HIVB>::default());
    normalised_template(SubstModel::<BLOSUM>::default());
}

#[test]
fn dna_normalised() {
    normalised_template(SubstModel::<JC69>::default());
    normalised_template(SubstModel::<K80>::default());
    normalised_template(SubstModel::<HKY>::default());
    normalised_template(SubstModel::<TN93>::default());
    normalised_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn setup_simple_phylo_info(blen_i: f64, blen_j: f64) -> PhyloInfo<MSA> {
    let tree = tree!(format!("((A0:{blen_i},B1:{blen_j}):1.0);").as_str());
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![record!("A0", b"A"), record!("B1", b"A")]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn dna_simple_likelihood() {
    let info = setup_simple_phylo_info(1.0, 1.0);
    let jc69 = SubstModel::<JC69>::default();
    let c = SCB::new(jc69, info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.5832498829317445, epsilon = 1e-6);

    let info = setup_simple_phylo_info(1.0, 2.0);
    let jc69 = SubstModel::<JC69>::default();
    let c = SCB::new(jc69, info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.719098272533848, epsilon = 1e-6);
}

#[cfg(test)]
fn setup_cb_example_phylo_info() -> PhyloInfo<MSA> {
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("one", b"C"),
            record!("two", b"A"),
            record!("three", b"T"),
            record!("four", b"G"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn change_logl_on_freq_change_template<Q: QMatrix>(model: SubstModel<Q>) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let mut c = SCB::new(model, info).build().unwrap();

    let logl = c.cost();
    let _ = c.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_ne!(logl, c.cost());
}

#[test]
fn change_logl_on_freq_change() {
    change_logl_on_freq_change_template(
        SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5; HKY_PARAM_N]).unwrap(),
    );
    change_logl_on_freq_change_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5; TN93_PARAM_N]).unwrap(),
    );
    change_logl_on_freq_change_template(
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5; GTR_PARAM_N]).unwrap(),
    );
}

#[cfg(test)]
fn same_logl_on_freq_change_template<Q: QMatrix>(model: SubstModel<Q>) {
    // likelihood should not change when frequencies are changed in models with fixed freqs
    let info = setup_cb_example_phylo_info();
    let mut c = SCB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert!(c.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4])).is_err());
    assert_eq!(logl, c.cost());
}

#[test]
fn same_logl_on_freq_change() {
    same_logl_on_freq_change_template(SubstModel::<JC69>::default());
    same_logl_on_freq_change_template(SubstModel::<K80>::default());
}

#[cfg(test)]
fn change_logl_on_param_change_template<Q: QMatrix>(model: SubstModel<Q>) {
    // likelihood should change when parameters are changed
    let info = setup_cb_example_phylo_info();
    let mut c = SCB::new(model, info).build().unwrap();

    for i in 0..c.model.params().len() {
        let logl = c.cost();
        c.set_param(i, 100.0);
        assert_ne!(logl, c.cost());
    }
}

#[test]
fn change_logl_on_param_change() {
    change_logl_on_param_change_template(SubstModel::<K80>::default());
    change_logl_on_param_change_template(SubstModel::<HKY>::default());
    change_logl_on_param_change_template(SubstModel::<TN93>::default());
    change_logl_on_param_change_template(SubstModel::<GTR>::default());
}

#[test]
fn same_likelihood_on_param_change() {
    // likelihood should not change when parameters are changed for jc69
    let info = setup_cb_example_phylo_info();
    let model = SubstModel::<JC69>::default();
    let mut c = SCB::new(model, info).build().unwrap();
    let logl = c.cost();
    c.set_param(0, 100.0);
    assert_eq!(logl, c.cost());
}

#[cfg(test)]
fn dna_gaps_as_ambigs_template<Q: QMatrix>(model: SubstModel<Q>) {
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("one", b"CCCCCCXX"),
            record!("two", b"XXAAAAAA"),
            record!("three", b"TTTNNTTT"),
            record!("four", b"GNGGGGNG"),
        ]),
        &tree,
    )
    .unwrap();
    let info_ambig = PhyloInfo {
        msa,
        tree: tree.clone(),
    };
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("one", b"CCCCCC--"),
            record!("two", b"--AAAAAA"),
            record!("three", b"TTT--TTT"),
            record!("four", b"G-GGGG-G"),
        ]),
        &tree,
    )
    .unwrap();
    let info_gaps = PhyloInfo { msa, tree };

    let c_ambig = SCB::new(model.clone(), info_ambig).build().unwrap();
    let c_gaps = SCB::new(model, info_gaps).build().unwrap();
    assert_eq!(c_ambig.cost(), c_gaps.cost());
}

#[test]
fn dna_gaps_as_ambigs() {
    dna_gaps_as_ambigs_template(SubstModel::<JC69>::default());
    dna_gaps_as_ambigs_template(SubstModel::<K80>::default());
    dna_gaps_as_ambigs_template(SubstModel::<HKY>::default());
    dna_gaps_as_ambigs_template(SubstModel::<TN93>::default());
    dna_gaps_as_ambigs_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn setup_phylo_info_single_leaf() -> PhyloInfo<MSA> {
    let tree = tree!("(A0:1.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![record!("A0", b"AAAAAA")]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn dna_likelihood_one_node_template<Q: QMatrix>(model: SubstModel<Q>) {
    let info = setup_phylo_info_single_leaf();
    let c = SCB::new(model, info).build().unwrap();
    assert!(c.cost() < 0.0);
}

#[test]
fn dna_likelihood_one_node() {
    dna_likelihood_one_node_template(SubstModel::<JC69>::default());
    dna_likelihood_one_node_template(SubstModel::<K80>::default());
    dna_likelihood_one_node_template(SubstModel::<HKY>::default());
    dna_likelihood_one_node_template(SubstModel::<TN93>::default());
    dna_likelihood_one_node_template(SubstModel::<GTR>::default());
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let mut model = SubstModel::<TN93>::default();
    // pre-computed unnormalised Q matrix for math check
    let _ = model.set_freqs(frequencies!(&[0.22, 0.26, 0.33, 0.19]));
    model.qmatrix.q = RateMatrix::from_row_slice(
        4,
        4,
        &[
            -0.15594579,
            0.15524379,
            0.00044550000000000004,
            0.0002565,
            0.13136013,
            -0.13206213,
            0.00044550000000000004,
            0.0002565,
            0.000297,
            0.000351,
            -0.056516265,
            0.055868265,
            0.000297,
            0.000351,
            0.097034355,
            -0.097682355,
        ],
    );
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), -17.1035117087, epsilon = 1e-6);
}

#[cfg(test)]
fn setup_mol_evo_example_phylo_info() -> PhyloInfo<MSA> {
    let tree = tree!("(((one:0.2,two:0.2):0.1,three:0.2):0.1,(four:0.2,five:0.2):0.1);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("one", b"T"),
            record!("two", b"C"),
            record!("three", b"A"),
            record!("four", b"C"),
            record!("five", b"C"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn dna_mol_evo_example_likelihood() {
    let info = setup_mol_evo_example_phylo_info();
    let model = SubstModel::<K80>::default();
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), -7.581408, epsilon = 1e-6);
}

#[cfg(test)]
fn dna_ambig_example_logl_template<Q: QMatrix>(model: SubstModel<Q>) {
    // Checks that likelihoods for different ambiguous characters are the same
    let fldr = Path::new("./data");
    let info_w_x = PIB::with_attrs(
        fldr.join("ambiguous_example.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let info_w_n = PIB::with_attrs(
        fldr.join("ambiguous_example_N.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();

    let c_w_x = SCB::new(model.clone(), info_w_x).build().unwrap();
    let c_w_n = SCB::new(model, info_w_n).build().unwrap();

    assert_relative_eq!(c_w_x.cost(), c_w_n.cost());
}

#[test]
fn dna_ambig_example_likelihood() {
    dna_ambig_example_logl_template(SubstModel::<JC69>::default());
    dna_ambig_example_logl_template(SubstModel::<K80>::default());
    dna_ambig_example_logl_template(SubstModel::<HKY>::default());
    dna_ambig_example_logl_template(SubstModel::<TN93>::default());
    dna_ambig_example_logl_template(SubstModel::<GTR>::default());
}

#[test]
fn dna_ambig_example_likelihood_k80() {
    // Checks the exact value for k80
    let fldr = Path::new("./data");
    let info_w_x = PIB::with_attrs(
        fldr.join("ambiguous_example.fasta"),
        fldr.join("ambiguous_example.newick"),
    )
    .build()
    .unwrap();
    let k80 = SubstModel::<K80>::default();
    let c = SCB::new(k80, info_w_x).build().unwrap();
    assert_relative_eq!(c.cost(), -137.24280493914029, epsilon = 1e-6);
}

#[test]
fn dna_huelsenbeck_example_likelihood() {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let hky = SubstModel::<HKY>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0]).unwrap();
    let c = SCB::new(hky, info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -216.234734, epsilon = 1e-3);
    let gtr_as_hky =
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.0, 0.2, 0.2, 0.2, 0.2]).unwrap();
    let c_gtr = SCB::new(gtr_as_hky, info).build().unwrap();
    assert_relative_eq!(c_gtr.cost(), -216.234734, epsilon = 1e-3);
}

#[cfg(test)]
fn protein_example_logl_template<Q: QMatrix>(
    model: SubstModel<Q>,
    expected_llik: f64,
    epsilon: f64,
) {
    let fldr = Path::new("./data/phyml_protein_example");
    let info = PIB::with_attrs(
        fldr.join("nogap_seqs.fasta"),
        fldr.join("example_tree.newick"),
    )
    .build()
    .unwrap();
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), expected_llik, epsilon = epsilon);
}

#[test]
fn protein_example_likelihood() {
    protein_example_logl_template(SubstModel::<WAG>::default(), -4505.736814460457, 1e-3);
    protein_example_logl_template(SubstModel::<HIVB>::default(), -4407.989226397638, 1e-5);
    protein_example_logl_template(SubstModel::<BLOSUM>::default(), -4587.71053, 1e-5);
}

#[cfg(test)]
fn simple_reroot_info(alphabet: &'static Alphabet) -> (PhyloInfo<MSA>, PhyloInfo<MSA>) {
    let tree = tree!("((A:2.0,B:2.0):1.0,C:2.0):0.0;");
    let seqs = Sequences::with_alphabet_unchecked(
        vec![
            record!("A", b"CTATATATACIJL"),
            record!("B", b"ATATATATAAIHL"),
            record!("C", b"TTATATATATIJL"),
        ],
        alphabet,
    );

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let tree_rerooted = tree!("(A:1.0,(B:2.0,C:3.0):1.0):0.0;");
    let info_rerooted = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree_rerooted).unwrap(),
        tree: tree_rerooted,
    };

    (info, info_rerooted)
}

#[cfg(test)]
fn logl_revers_template<Q: QMatrix>(model: SubstModel<Q>) {
    let (info, info_rerooted) = simple_reroot_info(Q::alphabet());

    let c = SCB::new(model.clone(), info).build().unwrap();
    let c_rerooted = SCB::new(model, info_rerooted).build().unwrap();
    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = 1e-10);
}

#[test]
fn dna_logl_reversibility() {
    logl_revers_template(SubstModel::<JC69>::default());
    logl_revers_template(SubstModel::<K80>::default());
    logl_revers_template(SubstModel::<HKY>::default());
    logl_revers_template(SubstModel::<TN93>::default());
    logl_revers_template(SubstModel::<GTR>::default());
}

#[test]
fn protein_logl_reversibility() {
    logl_revers_template(SubstModel::<WAG>::default());
    logl_revers_template(SubstModel::<HIVB>::default());
    logl_revers_template(SubstModel::<BLOSUM>::default());
}

fn huelsenbeck_reversibility_template<Q: QMatrix>(model: SubstModel<Q>) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();
    let info_rerooted = PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example_reroot.newick"),
    )
    .build()
    .unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let c_rerooted = SCB::new(model, info_rerooted).build().unwrap();
    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = 1e-10,);
}

#[test]
fn huelsenbeck_reversibility() {
    huelsenbeck_reversibility_template(SubstModel::<JC69>::default());
    huelsenbeck_reversibility_template(SubstModel::<K80>::default());
    huelsenbeck_reversibility_template(SubstModel::<HKY>::default());
    huelsenbeck_reversibility_template(SubstModel::<TN93>::default());
    huelsenbeck_reversibility_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn logl_correct_w_diff_info<Q: QMatrix>(model: SubstModel<Q>, llik1: f64, llik2: f64) {
    let tree1 = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let tree2 = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"P"),
        record!("B", b"P"),
        record!("C", b"P"),
        record!("D", b"P"),
    ]);

    let info1 = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree1).unwrap(),
        tree: tree1,
    };
    let info2 = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree2).unwrap(),
        tree: tree2,
    };

    let c1 = SCB::new(model.clone(), info1).build().unwrap();
    let c2 = SCB::new(model, info2).build().unwrap();

    assert_relative_eq!(c1.cost(), llik1, epsilon = 1e-5);
    assert_relative_eq!(c2.cost(), llik2, epsilon = 1e-5);
}

#[test]
fn protein_logl_correct_w_diff_info() {
    logl_correct_w_diff_info(
        SubstModel::<WAG>::default(),
        -7.488595394504073,
        -10.206456536551775,
    );
    logl_correct_w_diff_info(
        SubstModel::<HIVB>::default(),
        -7.231597482410509,
        -9.865559545434952,
    );
    logl_correct_w_diff_info(
        SubstModel::<BLOSUM>::default(),
        -7.4408154253528975,
        -10.33187874481282,
    );
}

#[cfg(test)]
fn one_site_one_char_template<Q: QMatrix>(model: SubstModel<Q>) {
    // This used to fail on leaf data creation when some of the sequences were empty
    let sequences = Sequences::with_alphabet_unchecked(
        vec![
            record!("one", b"C"),
            record!("two", b"-"),
            record!("three", b"-"),
            record!("four", b"-"),
        ],
        Q::alphabet(),
    );
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let info = PhyloInfo {
        msa: MSA::from_aligned(sequences, &tree).unwrap(),
        tree,
    };

    let c = SCB::new(model, info).build().unwrap();

    assert_ne!(c.cost(), f64::NEG_INFINITY);
    assert!(c.cost() < 0.0);
}

#[test]
fn dna_one_site_one_char() {
    one_site_one_char_template(SubstModel::<JC69>::default());
    one_site_one_char_template(SubstModel::<K80>::default());
    one_site_one_char_template(SubstModel::<HKY>::default());
    one_site_one_char_template(SubstModel::<TN93>::default());
    one_site_one_char_template(SubstModel::<GTR>::default());
}

#[test]
fn protein_one_site_one_char() {
    one_site_one_char_template(SubstModel::<WAG>::default());
    one_site_one_char_template(SubstModel::<HIVB>::default());
    one_site_one_char_template(SubstModel::<BLOSUM>::default());
}

#[test]
fn dna_gaps_against_phyml() {
    let tree =
        tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);");
    let seqs = Sequences::new_unchecked(
        read_sequences(Path::new("./data/").join("sequences_DNA1.fasta")).unwrap(),
    );
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree,
    };
    let jc69 = SubstModel::<JC69>::default();
    let c = SCB::new(jc69, info).build().unwrap();

    // Compare against value from PhyML
    assert_relative_eq!(c.cost(), -9.70406054783923);
}

#[test]
fn dna_single_char_gaps_against_phyml() {
    let jc69 = SubstModel::<JC69>::default();
    let tree =
        tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.000000:0.08716381);");

    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"A"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree: tree.clone(),
    };
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.920437792326963); // Compare against PhyML

    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"X"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree: tree.clone(),
    };

    let c = SCB::new(jc69.clone(), info).build().unwrap();
    assert_relative_eq!(c.cost(), 0.0, epsilon = 1e-10); // Compare against PhyML

    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"X"),
        record!("B", b"X"),
        record!("C", b"X"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree: tree.clone(),
    };
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    assert_relative_eq!(c.cost(), -1.38629, epsilon = 1e-5); // Compare against PhyML

    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"-"),
        record!("B", b"-"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree: tree.clone(),
    };
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.77259, epsilon = 1e-5); // Compare against PhyML

    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"-"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree,
    };
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.92044, epsilon = 1e-5); // Compare against PhyML
}

#[test]
fn dna_ambig_chars_against_phyml() {
    let jc69 = SubstModel::<JC69>::default();
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001)0.0:0.08716381);");
    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"B"),
        record!("B", b"A"),
        record!("C", b"A"),
        record!("D", b"T"),
    ]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree,
    };
    let c = SCB::new(jc69, info).build().unwrap();
    assert_relative_eq!(c.cost(), -21.28936836, epsilon = 1e-7);
}

#[test]
fn dna_x_simple_fully_likely() {
    let jc69 = SubstModel::<JC69>::default();
    let tree = tree!("(A:0.05,B:0.0005):0.0;");
    let seqs = Sequences::new_unchecked(vec![record!("A", b"X"), record!("B", b"X")]);
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree,
    };
    let c = SCB::new(jc69.clone(), info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), 0.0, epsilon = 1e-15);
}

#[cfg(test)]
fn x_fully_likely_template<Q: QMatrix>(model: SubstModel<Q>) {
    let tree = tree!("(((A:2.0,B:2.0)E:4.0,(C:2.0,D:2.0)F:4.0)G:6.0);");
    let seqs = Sequences::with_alphabet_unchecked(
        vec![
            record!("A", b"X"),
            record!("B", b"X"),
            record!("C", b"X"),
            record!("D", b"X"),
        ],
        Q::alphabet(),
    );
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs, &tree).unwrap(),
        tree,
    };
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), 0.0, epsilon = 1e-5);
}

#[test]
fn dna_x_fully_likely() {
    x_fully_likely_template(SubstModel::<JC69>::default());
    x_fully_likely_template(SubstModel::<K80>::default());
    x_fully_likely_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap());
    x_fully_likely_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    );
    x_fully_likely_template(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[test]
fn protein_x_fully_likely() {
    x_fully_likely_template(SubstModel::<WAG>::default());
    x_fully_likely_template(SubstModel::<HIVB>::default());
    x_fully_likely_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn avg_rate_template<Q: QMatrix>(model: SubstModel<Q>) {
    let avg_rate = model.q().diagonal().component_mul(model.freqs()).sum();
    assert_relative_eq!(avg_rate, -1.0, epsilon = 1e-10);
}

#[test]
fn dna_avg_rate() {
    avg_rate_template(SubstModel::<JC69>::default());
    avg_rate_template(SubstModel::<K80>::default());
    avg_rate_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap());
    avg_rate_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    );
    avg_rate_template(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[test]
fn protein_avg_rate() {
    avg_rate_template(SubstModel::<WAG>::default());
    avg_rate_template(SubstModel::<HIVB>::default());
    avg_rate_template(SubstModel::<BLOSUM>::default());
    let freqs = &[1.0 / 20.0; 20];
    avg_rate_template(SubstModel::<WAG>::new(freqs, &[]).unwrap());
    avg_rate_template(SubstModel::<HIVB>::new(freqs, &[]).unwrap());
    avg_rate_template(SubstModel::<BLOSUM>::new(freqs, &[]).unwrap());
}

#[cfg(test)]
fn p_matrix_limit_template<Q: QMatrix>(model: SubstModel<Q>) {
    let time = 1e10f64;
    let p = model.p(time);
    let p2 = model.p(1e3f64);
    check_freq_convergence(p.clone(), model.freqs(), 1e-5);
    check_freq_convergence(p2.clone(), model.freqs(), 1e-5);
    assert_relative_eq!(p, p2, epsilon = 1e-5);
}

#[test]
fn p_matrix_limit() {
    p_matrix_limit_template(SubstModel::<WAG>::default());
    p_matrix_limit_template(SubstModel::<HIVB>::default());
    p_matrix_limit_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn parsimony_rounding_template<Q: QMatrix>(model: SubstModel<Q>) {
    let mat_round = model.scoring(0.1, &Z::zero(), &R::zero());
    let mat = model.scoring(0.1, &Z::zero(), &R::none());
    assert_ne!(mat_round.mean(), mat.mean());
    assert_ne!(mat_round, mat);
    for (&e1, &e2) in mat_round.as_slice().iter().zip(mat.as_slice().iter()) {
        assert_relative_eq!(e1, e1.round());
        assert_relative_eq!(e1, e2.round());
    }
}

#[test]
fn protein_rounding_scores() {
    parsimony_rounding_template(SubstModel::<HIVB>::default());
    parsimony_rounding_template(SubstModel::<WAG>::default());
    parsimony_rounding_template(SubstModel::<BLOSUM>::default());
}

#[test]
fn dna_rounding_scores() {
    parsimony_rounding_template(SubstModel::<JC69>::default());
    parsimony_rounding_template(SubstModel::<K80>::default());
    parsimony_rounding_template(SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap());
    parsimony_rounding_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    );
    parsimony_rounding_template(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[cfg(test)]
fn parsimony_zero_diag_template<Q: QMatrix>(model: SubstModel<Q>) {
    let mat_zeros = model.scoring(0.1, &Z::zero(), &R::none());
    let mat = model.scoring(0.1, &Z::non_zero(), &R::none());
    assert_ne!(mat_zeros.mean(), mat.mean());
    assert_ne!(mat_zeros, mat);
    for (&e1, &e2) in mat_zeros.diagonal().iter().zip(mat.diagonal().iter()) {
        assert_relative_eq!(e1, 0.0);
        assert_ne!(e1, e2);
        assert_ne!(e2, 0.0);
    }
}

#[test]
fn protein_zero_diag_scores() {
    parsimony_zero_diag_template(SubstModel::<HIVB>::default());
    parsimony_zero_diag_template(SubstModel::<WAG>::default());
    parsimony_zero_diag_template(SubstModel::<BLOSUM>::default());
}

#[test]
fn dna_zero_diag_scores() {
    parsimony_zero_diag_template(SubstModel::<JC69>::default());
    parsimony_zero_diag_template(SubstModel::<K80>::default());
    parsimony_zero_diag_template(
        SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap(),
    );
    parsimony_zero_diag_template(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    );
    parsimony_zero_diag_template(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[cfg(test)]
fn setup_test_info(alphabet: &'static Alphabet) -> PhyloInfo<MSA> {
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![
                record!("A", b"CTATATATAC"),
                record!("B", b"ATATATATAA"),
                record!("C", b"TTATATATAT"),
                record!("D", b"TTATATATAT"),
            ],
            alphabet,
        ),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn dirty_tree_costs_match_template<Q: QMatrix>(model: SubstModel<Q>) {
    use crate::likelihood::TreeSearchCost;

    let info = setup_test_info(Q::alphabet());

    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = TreeSearchCost::cost(&c);
    assert_eq!(logl, TreeSearchCost::cost(&c));

    // The likelihood should be the same if we make the tree dirty without changing it
    let mut tree = c.info.tree.clone();
    tree.dirty();
    c.update_tree(tree);
    assert_eq!(logl, TreeSearchCost::cost(&c));

    // The likelihood should be the same if we rebuild from scratch
    let c2 = SCB::new(model, info).build().unwrap();
    let logl2 = TreeSearchCost::cost(&c2);
    assert_eq!(logl2, TreeSearchCost::cost(&c2));
    assert_eq!(logl, logl2);
}

#[test]
fn dirty_tree_costs_match() {
    dirty_tree_costs_match_template(SubstModel::<JC69>::default());
    dirty_tree_costs_match_template(SubstModel::<K80>::default());
    dirty_tree_costs_match_template(SubstModel::<HKY>::default());
    dirty_tree_costs_match_template(SubstModel::<TN93>::default());
    dirty_tree_costs_match_template(SubstModel::<GTR>::default());

    dirty_tree_costs_match_template(SubstModel::<WAG>::default());
    dirty_tree_costs_match_template(SubstModel::<HIVB>::default());
    dirty_tree_costs_match_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn dirty_branch_costs_match_template<Q: QMatrix>(model: SubstModel<Q>) {
    use crate::likelihood::TreeSearchCost;

    let info = setup_test_info(Q::alphabet());

    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = TreeSearchCost::cost(&c);

    // The likelihood should change if we change branch lengths
    let mut mutated_tree = info.tree.clone();
    mutated_tree.set_blen(&info.tree.by_id("F").idx, 4.0);
    mutated_tree.set_blen(&info.tree.by_id("B").idx, 0.75);
    c.update_tree(mutated_tree);

    let logl2 = TreeSearchCost::cost(&c);
    assert_eq!(logl2, TreeSearchCost::cost(&c));
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let new_tree = tree!("(((A:1.0,B:0.75)E:2.0,(C:1.0,D:1.0)F:4.0)G:3.0);");
    let new_info = PhyloInfo {
        msa: info.msa.clone(),
        tree: new_tree,
    };
    let c = SCB::new(model, new_info).build().unwrap();
    let new_logl = TreeSearchCost::cost(&c);
    assert_eq!(new_logl, TreeSearchCost::cost(&c));
    assert_eq!(logl2, new_logl);
}

#[test]
fn dirty_branch_costs_match() {
    dirty_branch_costs_match_template(SubstModel::<JC69>::default());
    dirty_branch_costs_match_template(SubstModel::<K80>::default());
    dirty_branch_costs_match_template(SubstModel::<HKY>::default());
    dirty_branch_costs_match_template(SubstModel::<TN93>::default());
    dirty_branch_costs_match_template(SubstModel::<GTR>::default());

    dirty_branch_costs_match_template(SubstModel::<WAG>::default());
    dirty_branch_costs_match_template(SubstModel::<HIVB>::default());
    dirty_branch_costs_match_template(SubstModel::<BLOSUM>::default());
}

#[cfg(test)]
fn modify_model_params_costs_match_template<Q: QMatrix + QMatrixMaker>(model: SubstModel<Q>) {
    let info = setup_test_info(Q::alphabet());

    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = c.cost();

    // The likelihood should change if we change model parameters
    c.set_param(0, 0.5);

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let mut new_params = model.params().to_vec();
    new_params[0] = 0.5;
    let new_model = SubstModel::<Q>::new(model.freqs().as_slice(), &new_params).unwrap();
    let c = SCB::new(new_model, info).build().unwrap();
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_params_costs_match() {
    // does not apply to JC69, WAG, HIVB, BLOSUM which have no params
    modify_model_params_costs_match_template(SubstModel::<K80>::default());
    modify_model_params_costs_match_template(SubstModel::<HKY>::default());
    modify_model_params_costs_match_template(SubstModel::<TN93>::default());
    modify_model_params_costs_match_template(SubstModel::<GTR>::default());
}

#[cfg(test)]
fn modify_model_freqs_costs_match_template<Q: QMatrix + QMatrixMaker>(
    model: SubstModel<Q>,
    freqs: FreqVector,
) {
    let info = setup_test_info(Q::alphabet());

    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = c.cost();

    // The likelihood should change if we change model frequencies
    assert!(c.set_freqs(freqs.clone()).is_ok());

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let new_model = SubstModel::<Q>::new(freqs.as_slice(), model.params()).unwrap();
    let c = SCB::new(new_model, info).build().unwrap();
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_freqs_costs_match() {
    // does not apply to JC69 and K80 which have no freqs
    let new_dna_freqs = frequencies!(&[0.1, 0.1, 0.1, 0.7]);
    modify_model_freqs_costs_match_template(SubstModel::<HKY>::default(), new_dna_freqs.clone());
    modify_model_freqs_costs_match_template(SubstModel::<TN93>::default(), new_dna_freqs.clone());
    modify_model_freqs_costs_match_template(SubstModel::<GTR>::default(), new_dna_freqs);

    let new_aa_freqs = frequencies!(&[0.05; 20]);
    modify_model_freqs_costs_match_template(SubstModel::<WAG>::default(), new_aa_freqs.clone());
    modify_model_freqs_costs_match_template(SubstModel::<BLOSUM>::default(), new_aa_freqs.clone());
    modify_model_freqs_costs_match_template(SubstModel::<HIVB>::default(), new_aa_freqs);
}

#[cfg(test)]
fn degenerate_freqs_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    let n = Q::alphabet().len();
    for i in 0..n {
        let mut degenerate_freqs = vec![0.0; n];
        degenerate_freqs[i] = 1.0;
        assert_matches!(
            SubstModel::<Q>::new(&degenerate_freqs, params),
            Err(Error::SubstitutionModel(
                SubstitutionModelError::DegenerateFrequencies { .. }
            ))
        );
    }
}

#[test]
fn protein_degenerate_freqs() {
    degenerate_freqs_template::<WAG>(&[]);
    degenerate_freqs_template::<HIVB>(&[]);
    degenerate_freqs_template::<BLOSUM>(&[]);
}

#[test]
fn dna_degenerate_freqs() {
    degenerate_freqs_template::<HKY>(&[2.0; HKY_PARAM_N]);
    degenerate_freqs_template::<TN93>(&[2.0; TN93_PARAM_N]);
    degenerate_freqs_template::<GTR>(&[2.0; GTR_PARAM_N]);
}

#[cfg(test)]
fn almost_degenerate_freqs_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    let n = Q::alphabet().len();
    for i in 0..n {
        let mut almost_degenerate_freqs = vec![f64::EPSILON; n];
        almost_degenerate_freqs[i] = 1.0 - (n as f64 - 1.0) * f64::EPSILON;
        let model = SubstModel::<Q>::new(&almost_degenerate_freqs, params).unwrap();
        assert!(model
            .q()
            .iter()
            .all(|&x| !x.is_nan() && x.abs() != f64::INFINITY));
    }
}

#[test]
fn almost_degenerate_dna_frequencies() {
    almost_degenerate_freqs_template::<HKY>(&[2.0; HKY_PARAM_N]);
    almost_degenerate_freqs_template::<TN93>(&[2.0; TN93_PARAM_N]);
    almost_degenerate_freqs_template::<GTR>(&[2.0; GTR_PARAM_N]);
}

#[test]
fn almost_degenerate_protein_frequencies() {
    almost_degenerate_freqs_template::<WAG>(&[]);
    almost_degenerate_freqs_template::<HIVB>(&[]);
    almost_degenerate_freqs_template::<BLOSUM>(&[]);
}

#[cfg(test)]
fn equal_freqs_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    let n = Q::alphabet().len();
    let equal_freqs = vec![1.0 / n as f64; n];
    assert!(SubstModel::<Q>::new(&equal_freqs, params).is_ok());
}

#[test]
fn equal_freqs_dna() {
    equal_freqs_template::<JC69>(&[]);
    equal_freqs_template::<K80>(&[2.0; K80_PARAM_N]);
    equal_freqs_template::<HKY>(&[2.0; HKY_PARAM_N]);
    equal_freqs_template::<TN93>(&[2.0; TN93_PARAM_N]);
    equal_freqs_template::<GTR>(&[2.0; GTR_PARAM_N]);
}

#[test]
fn equal_freqs_protein() {
    equal_freqs_template::<WAG>(&[]);
    equal_freqs_template::<HIVB>(&[]);
    equal_freqs_template::<BLOSUM>(&[]);
}
