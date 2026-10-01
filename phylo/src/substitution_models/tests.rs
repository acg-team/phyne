use std::ops::Mul;
use std::path::Path;

use approx::assert_relative_eq;
use itertools::repeat_n;
use nalgebra::dvector;
use rand::{rng, Rng};

use crate::alignment::{Alignment, Sequences, MSA};
use crate::alphabets::{Alphabet, AMINOACIDS, GAP};
use crate::evolutionary_models::EvoModel;
use crate::io::read_sequences;
use crate::likelihood::ModelSearchCost;
use crate::parsimony::{DiagonalZeros as Z, ParsimonyModel, Rounding as R};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, QMatrixMaker, SubstMatrix, SubstModel,
    SubstitutionCostBuilder as SCB,
};
use crate::{record_wo_desc as record, tree};

#[cfg(test)]
fn freqs_fixed_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    // freqs should not change for JC69 and K80
    let mut model = SubstModel::<Q>::new(&[], params).unwrap();
    assert!(model
        .set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]))
        .is_err());
    assert_eq!(model.freqs(), &frequencies!(&[0.25; 4]));
}

#[test]
fn dna_freqs_fixed() {
    freqs_fixed_template::<JC69>(&[]);
    freqs_fixed_template::<K80>(&[2.0]);
}

#[cfg(test)]
fn freqs_fixed_equal_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    // freqs should not change for JC69 and K80
    let mut model = SubstModel::<Q>::new(&[], params).unwrap();
    assert!(model
        .set_freqs(frequencies!(&[0.25, 0.25, 0.25, 0.25]))
        .is_ok());
    assert_eq!(model.freqs(), &frequencies!(&[0.25; 4]));
}

#[test]
fn dna_freqs_fixed_equal() {
    freqs_fixed_equal_template::<JC69>(&[]);
    freqs_fixed_equal_template::<K80>(&[2.0]);
}

#[cfg(test)]
fn freqs_updated_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    // freqs should change for HKY, TN93, and GTR
    let mut model = SubstModel::<Q>::new(freqs, params).unwrap();
    let new_freqs = frequencies!(&[0.1, 0.2, 0.3, 0.4]);
    assert!(model.set_freqs(new_freqs.clone()).is_ok());
    assert_eq!(model.freqs(), &new_freqs);
    assert_ne!(model.freqs(), &frequencies!(freqs));
}

#[test]
fn dna_freqs_updated() {
    freqs_updated_template::<HKY>(&[0.2, 0.1, 0.5, 0.2], &[2.0]);
    freqs_updated_template::<TN93>(&[0.1, 0.1, 0.44, 0.26], &[2.0, 1.0, 3.0]);
    freqs_updated_template::<GTR>(&[0.7, 0.1, 0.1, 0.1], &[2.0, 2.0, 2.0, 2.0, 2.0]);
}

#[cfg(test)]
fn freqs_not_updated_template<Q: QMatrix + QMatrixMaker>() {
    // freqs should not change when set to invalid values
    let mut model = SubstModel::<Q>::new(&[], &[]).unwrap();

    let correct_freqs = model.freqs().clone();
    assert!(model
        .set_freqs(frequencies!(&[-1.0, 1.5, -0.3, 0.8]))
        .is_err());

    assert_eq!(model.freqs(), &correct_freqs);
}

#[test]
fn dna_freqs_not_updated() {
    freqs_not_updated_template::<JC69>();
    freqs_not_updated_template::<K80>();
    freqs_not_updated_template::<HKY>();
    freqs_not_updated_template::<TN93>();
    freqs_not_updated_template::<GTR>();
}

#[cfg(test)]
fn protein_freqs_not_updated_template<Q: QMatrix + QMatrixMaker>() {
    // freqs should not change when set to invalid values
    let mut model = SubstModel::<Q>::new(&[], &[]).unwrap();

    let correct_freqs = model.freqs().clone();
    assert!(model.set_freqs(frequencies!(&[-0.1; 20])).is_err());

    assert_eq!(model.freqs(), &correct_freqs);
}

#[test]
fn protein_freqs_not_updated() {
    protein_freqs_not_updated_template::<WAG>();
    protein_freqs_not_updated_template::<HIVB>();
    protein_freqs_not_updated_template::<BLOSUM>();
}

#[cfg(test)]
fn param_fixed_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    // parameters should not change for JC69
    let model = SubstModel::<Q>::new(&[], params).unwrap();
    let mut updated_model = model.clone();
    updated_model.set_param(0, 66.0);
    assert_eq!(model.params(), updated_model.params());
    assert!(updated_model.params().is_empty());
}

#[test]
fn dna_params_fixed() {
    param_fixed_template::<JC69>(&[]);
}

#[cfg(test)]
fn params_updated_template<Q: QMatrix + QMatrixMaker>(params: &[f64], new_params: &[f64]) {
    // parameters should change for K80, HKY, TN93, and GTR
    let mut model = SubstModel::<Q>::new(&[], params).unwrap();
    for (i, &param) in new_params.iter().enumerate() {
        model.set_param(i, param);
    }
    assert_eq!(model.params(), new_params);
}

#[test]
fn dna_params_updated() {
    params_updated_template::<HKY>(&[0.2], &[2.0]);
    params_updated_template::<TN93>(&[0.1, 0.1, 0.44], &[2.0, 1.0, 3.0]);
    params_updated_template::<GTR>(&[0.7; 5], &[0.9; 5]);
}

#[cfg(test)]
fn check_freq_convergence(substmat: SubstMatrix, pi: &FreqVector, epsilon: f64) {
    assert_eq!(substmat.nrows(), pi.len());
    assert_eq!(substmat.ncols(), pi.len());
    for row in substmat.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = epsilon);
        assert_relative_eq!(row, pi.transpose().as_view(), epsilon = epsilon);
    }
}

#[test]
fn dna_jc69_correct() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let jc69_2 = SubstModel::<JC69>::new(&[], &[1.0, 2.0]).unwrap();
    assert_eq!(jc69, jc69_2);
    assert_relative_eq!(jc69.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(jc69.rate(b'A', b'C'), 1.0 / 3.0);
    assert_relative_eq!(jc69.rate(b'G', b'T'), 1.0 / 3.0);
    assert_relative_eq!(jc69.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    let jc69_3 = SubstModel::<JC69>::new(&[], &[4.0]).unwrap();
    assert_eq!(jc69.q(), jc69_3.q());
    assert_eq!(jc69.freqs(), jc69_3.freqs());
}

#[test]
fn dna_j69_params() {
    let jc69 = SubstModel::<JC69>::new(&[0.1, 0.4, 0.75, 1.5], &[0.1, 0.4, 0.75, 1.5]).unwrap();
    assert_relative_eq!(jc69.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(format!("{jc69}"), "JC69".to_string());
}

#[test]
fn dna_k80_correct() {
    let k80 = SubstModel::<K80>::new(&[], &[]).unwrap();
    let k801 = SubstModel::<K80>::new(&[], &[2.0]).unwrap();
    let k802 = SubstModel::<K80>::new(&[], &[2.0, 1.0]).unwrap();
    let k803 = SubstModel::<K80>::new(&[], &[2.0, 1.0, 3.0, 6.0]).unwrap();
    assert_eq!(k80, k801);
    assert_eq!(k80, k802);
    assert_eq!(k80, k803);
    assert_eq!(k802, k803);
    assert_relative_eq!(k80.rate(b'A', b'A'), -1.0);
    assert_relative_eq!(k80.rate(b'T', b'A'), 1.0 * 0.25);
    assert_relative_eq!(k80.rate(b'A', b'G'), 2.0 * 0.25);
    assert_relative_eq!(k80.freqs(), &dvector![0.25, 0.25, 0.25, 0.25]);
}

#[cfg(test)]
fn infinity_p_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model: SubstModel<Q> = SubstModel::<Q>::new(freqs, params).unwrap();
    let p_inf = model.p(1000.0);
    assert_eq!(p_inf.shape(), model.q().shape());
    check_freq_convergence(p_inf, model.freqs(), 1e-5);
}

#[test]
fn dna_infinity_p() {
    infinity_p_template::<JC69>(&[], &[]);
    infinity_p_template::<K80>(&[], &[]);
    infinity_p_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    infinity_p_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    infinity_p_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn protein_infinity_p() {
    infinity_p_template::<WAG>(&[], &[]);
    infinity_p_template::<HIVB>(&[], &[]);
    infinity_p_template::<BLOSUM>(&[], &[]);
}

#[test]
fn dna_k80_params() {
    let k80 = SubstModel::<K80>::new(&[0.1, 0.4, 0.75, 1.5], &[0.1, 0.4, 0.75, 1.5]).unwrap();
    assert_relative_eq!(k80.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(format!("{k80}"), format!("K80 with [kappa = {:.5}]", 0.1));
}

#[test]
fn dna_hky_incorrect() {
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[2.0]);
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[0.5]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[0.5]);
    let hky = SubstModel::<HKY>::new(&[2.0, 1.0, 3.0, 6.0], &[0.5, 1.0]).unwrap();
    assert_eq!(hky.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(hky.params(), &[0.5]);
}

#[test]
fn dna_hky_correct() {
    let hky = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap();
    assert_relative_eq!(hky.freqs(), &dvector![0.22, 0.26, 0.33, 0.19]);
    let hky2 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 1.0]).unwrap();
    assert_relative_eq!(hky2.freqs(), &dvector![0.22, 0.26, 0.33, 0.19]);
    assert_eq!(hky, hky2);
    let hky3 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[]).unwrap();
    let hky4 = SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[2.0, 1.0]).unwrap();
    assert_relative_eq!(
        hky3.q()
            .diagonal()
            .component_mul(&frequencies!(&[0.22, 0.26, 0.33, 0.19]))
            .sum(),
        -1.0
    );
    assert_eq!(hky3, hky4);
}

#[test]
fn dna_gtr_equal_rates_correct() {
    let gtr = SubstModel::<GTR>::new(
        &repeat_n(0.25, 4).collect::<Vec<f64>>(),
        &repeat_n(1.0, 5).collect::<Vec<f64>>(),
    )
    .unwrap();
    assert_eq!(gtr.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(gtr.q()[(0, 0)], -1.0);
    let gtr2 = SubstModel::<GTR>::new(&[], &repeat_n(1.0, 5).collect::<Vec<f64>>()).unwrap();
    assert_relative_eq!(gtr.q(), gtr2.q());
    assert!(gtr.rate(b'T', b'T') < 0.0);
    assert!(gtr.rate(b'A', b'A') < 0.0);
    assert_relative_eq!(gtr.rate(b'T', b'C'), gtr.rate(b'C', b'T'));
    assert_relative_eq!(gtr.rate(b'A', b'G'), gtr.rate(b'G', b'A'));
    assert_relative_eq!(gtr.freqs(), &frequencies!(&[0.25, 0.25, 0.25, 0.25]));
}

#[test]
fn dna_gtr_defaults() {
    let gtr_no_freqs = SubstModel::<GTR>::new(&[], &[2.0, 1.0, 3.0, 6.0, 0.5]).unwrap();
    assert_relative_eq!(gtr_no_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(gtr_no_freqs.params(), &[2.0, 1.0, 3.0, 6.0, 0.5]);

    let gtr_missing_params =
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.6, 0.7]).unwrap();
    assert_relative_eq!(
        gtr_missing_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(gtr_missing_params.params(), &[0.5, 0.6, 0.7, 1.0, 1.0]);

    let gtr_incorrect_freqs = SubstModel::<GTR>::new(&[0.3; 4], &[0.5; 5]).unwrap();
    assert_relative_eq!(gtr_incorrect_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(gtr_incorrect_freqs.params(), &[0.5; 5]);

    let gtr_too_many_params = SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7; 6]).unwrap();
    assert_relative_eq!(
        gtr_too_many_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(gtr_too_many_params.params(), &[0.7; 6]);
}

#[test]
fn dna_tn93_correct() {
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135])
        .unwrap();
    let expected_pi = frequencies!(&[0.22, 0.26, 0.33, 0.19]);
    let expected_q = SubstMatrix::from_column_slice(
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
fn dna_tn93_incorrect() {
    let tn93_too_few_params =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap();
    assert_relative_eq!(
        tn93_too_few_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(tn93_too_few_params.params(), &[0.5970915, 0.2940435, 1.0]);

    let tn93_incorrect_freqs =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 1.19], &[0.5, 0.6, 0.3]).unwrap();
    assert_relative_eq!(tn93_incorrect_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(tn93_incorrect_freqs.params(), &[0.5, 0.6, 0.3]);

    let tn93_too_many_params =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.6, 0.3, 0.56]).unwrap();
    assert_relative_eq!(
        tn93_too_many_params.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19])
    );
    assert_eq!(tn93_too_many_params.params(), &[0.5, 0.6, 0.3]);

    let tn93_no_freqs = SubstModel::<TN93>::new(&[], &[2.0, 1.0, 3.0]).unwrap();
    assert_relative_eq!(tn93_no_freqs.freqs(), &frequencies!(&[0.25; 4]));
    assert_eq!(tn93_no_freqs.params(), &[2.0, 1.0, 3.0]);
}

#[test]
fn dna_normalisation() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    assert_relative_eq!(jc69.q().diagonal().component_mul(jc69.freqs()).sum(), -1.0);
    let k80 = SubstModel::<K80>::new(&[], &[3.0, 1.5]).unwrap();
    assert_relative_eq!(k80.q().diagonal().component_mul(k80.freqs()).sum(), -1.0);
    let gtr = SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[0.7; 5]).unwrap();
    assert_relative_eq!(gtr.q().diagonal().component_mul(gtr.freqs()).sum(), -1.0);
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.59, 0.29, 0.0013]).unwrap();
    assert_relative_eq!(tn93.q().diagonal().component_mul(tn93.freqs()).sum(), -1.0);
}

#[test]
fn dna_normalised_param_change() {
    let mut k80 = SubstModel::<K80>::new(&[], &[3.0]).unwrap();
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
fn protein_correct_access_template<Q: QMatrix + QMatrixMaker>(epsilon: f64) {
    let model_1 = SubstModel::<Q>::new(&[], &[]).unwrap();
    let model_2 = SubstModel::<Q>::new(&[], &[]).unwrap();
    assert_relative_eq!(model_1.q(), model_2.q());
    for _ in 0..10 {
        let mut rng = rng();
        let query1 = AMINOACIDS[rng.random_range(0..AMINOACIDS.len())];
        let query2 = AMINOACIDS[rng.random_range(0..AMINOACIDS.len())];
        model_1.rate(query1, query2);
    }
    assert_relative_eq!(model_1.freqs().sum(), 1.0, epsilon = epsilon);
}

#[test]
fn protein_correct_access() {
    protein_correct_access_template::<WAG>(1e-4);
    protein_correct_access_template::<HIVB>(1e-3);
    protein_correct_access_template::<BLOSUM>(1e-3);
}

#[cfg(test)]
fn protein_gap_access_template<Q: QMatrix + QMatrixMaker>() {
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    model.rate(GAP, b'L');
}

#[test]
#[should_panic]
fn protein_incorrect_access_wag() {
    protein_gap_access_template::<WAG>();
}

#[test]
#[should_panic]
fn protein_incorrect_access_hivb() {
    protein_gap_access_template::<HIVB>();
}

#[test]
#[should_panic]
fn protein_incorrect_access_blosum() {
    protein_gap_access_template::<BLOSUM>();
}

#[cfg(test)]
fn normalised_template<Q: QMatrix + QMatrixMaker>() {
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    assert_relative_eq!(
        (model.q().diagonal().transpose().mul(model.freqs()))[(0, 0)],
        -1.0,
        epsilon = 1e-10
    );
}

#[test]
fn protein_normalised() {
    normalised_template::<WAG>();
    normalised_template::<HIVB>();
    normalised_template::<BLOSUM>();
}

#[test]
fn dna_normalised() {
    normalised_template::<JC69>();
    normalised_template::<K80>();
    normalised_template::<HKY>();
    normalised_template::<TN93>();
    normalised_template::<GTR>();
}

#[test]
fn designation() {
    let jc69_model_desc = format!("{}", SubstModel::<JC69>::new(&[], &[2.0]).unwrap());
    assert!(jc69_model_desc.contains("JC69"));
    assert!(!jc69_model_desc.contains("2.0"));

    let k80_model_desc = format!("{}", SubstModel::<K80>::new(&[], &[2.0]).unwrap());
    assert!(k80_model_desc.contains("K80"));
    assert!(k80_model_desc.contains("kappa = 2.0"));

    let hky_model_desc = format!("{}", SubstModel::<HKY>::new(&[], &[2.5]).unwrap());
    assert!(hky_model_desc.contains("HKY"));
    assert!(hky_model_desc.contains("kappa = 2.5"));

    let tn93_model_desc = format!(
        "{}",
        SubstModel::<TN93>::new(&[], &[2.5, 0.3, 0.1]).unwrap()
    );
    assert!(tn93_model_desc.contains("TN93"));
    assert!(tn93_model_desc.contains("2.5"));
    assert!(tn93_model_desc.contains("0.3"));
    assert!(tn93_model_desc.contains("0.1"));

    let gtr_model_desc = format!(
        "{}",
        SubstModel::<GTR>::new(&[0.22, 0.26, 0.33, 0.19], &[1.5, 3.0, 1.25, 0.45, 0.1]).unwrap()
    );
    assert!(gtr_model_desc.contains("GTR"));
    assert!(gtr_model_desc.contains("rtc = 1.5"));
    assert!(gtr_model_desc.contains("rta = 3.0"));
    assert!(gtr_model_desc.contains("rtg = 1.25"));
    assert!(gtr_model_desc.contains("rca = 0.45"));
    assert!(gtr_model_desc.contains("rcg = 0.1"));
    assert!(gtr_model_desc.contains("rag = 1.0"));
    assert!(gtr_model_desc.contains("0.22"));
    assert!(gtr_model_desc.contains("0.26"));
    assert!(gtr_model_desc.contains("0.33"));
    assert!(gtr_model_desc.contains("0.19"));

    let wag_model_desc = format!("{}", SubstModel::<WAG>::new(&[], &[]).unwrap());
    assert!(wag_model_desc.contains("WAG"));

    let hivb_model_desc = format!("{}", SubstModel::<HIVB>::new(&[], &[]).unwrap());
    assert!(hivb_model_desc.contains("HIVB"));

    let blosum_model_desc = format!("{}", SubstModel::<BLOSUM>::new(&[], &[]).unwrap());
    assert!(blosum_model_desc.contains("BLOSUM"));
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
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69, info).build().unwrap();
    assert_relative_eq!(c.cost(), -2.5832498829317445, epsilon = 1e-6);

    let info = setup_simple_phylo_info(1.0, 2.0);
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
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
fn change_logl_on_freq_change_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let mut c = SCB::new(model, info).build().unwrap();

    let logl = c.cost();
    c.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4])).unwrap();
    assert_ne!(logl, c.cost());
}

#[test]
fn change_logl_on_freq_change() {
    change_logl_on_freq_change_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_freq_change_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_freq_change_template::<GTR>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
}

#[cfg(test)]
fn same_logl_on_freq_change_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let mut c = SCB::new(model, info).build().unwrap();
    let logl = c.cost();
    let _ = c.set_freqs(frequencies!(&[0.1, 0.2, 0.3, 0.4]));
    assert_eq!(logl, c.cost());
}

#[test]
fn same_logl_on_freq_change() {
    same_logl_on_freq_change_template::<JC69>(&[0.22, 0.26, 0.33, 0.19], &[2.0]);
    same_logl_on_freq_change_template::<K80>(&[0.22, 0.26, 0.33, 0.19], &[]);
}

#[cfg(test)]
fn change_logl_on_param_change_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    // likelihood should change when frequencies are changed in models with free freqs
    let info = setup_cb_example_phylo_info();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let mut c = SCB::new(model, info).build().unwrap();
    let logl = c.cost();
    c.set_param(0, 100.0);
    assert_ne!(logl, c.cost());
}

#[test]
fn change_logl_on_param_change() {
    change_logl_on_param_change_template::<K80>(&[], &[2.0]);
    change_logl_on_param_change_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    change_logl_on_param_change_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    change_logl_on_param_change_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    );
}

#[test]
fn same_likelihood_on_param_change() {
    // likelihood should not change when parameters are changed for jc69
    let info = setup_cb_example_phylo_info();
    let model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let mut c = SCB::new(model, info).build().unwrap();
    let logl = c.cost();
    c.set_param(0, 100.0);
    assert_eq!(logl, c.cost());
}

#[cfg(test)]
fn dna_gaps_as_ambigs_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
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

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let c_ambig = SCB::new(model.clone(), info_ambig).build().unwrap();
    let c_gaps = SCB::new(model, info_gaps).build().unwrap();
    assert_eq!(c_ambig.cost(), c_gaps.cost());
}

#[test]
fn dna_gaps_as_ambigs() {
    dna_gaps_as_ambigs_template::<JC69>(&[], &[]);
    dna_gaps_as_ambigs_template::<K80>(&[], &[]);
    dna_gaps_as_ambigs_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    dna_gaps_as_ambigs_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    dna_gaps_as_ambigs_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
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
fn dna_likelihood_one_node_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let info = setup_phylo_info_single_leaf();
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let c = SCB::new(model, info).build().unwrap();
    assert!(c.cost() < 0.0);
}

#[test]
fn dna_likelihood_one_node() {
    dna_likelihood_one_node_template::<JC69>(&[], &[]);
    dna_likelihood_one_node_template::<K80>(&[], &[]);
    dna_likelihood_one_node_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    dna_likelihood_one_node_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    dna_likelihood_one_node_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn dna_cb_example_likelihood() {
    let info = setup_cb_example_phylo_info();
    let mut model =
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135])
            .unwrap();
    model.qmatrix.q = SubstMatrix::from_row_slice(
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
    let model = SubstModel::<K80>::new(&[], &[]).unwrap();
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), -7.581408, epsilon = 1e-6);
}

#[cfg(test)]
fn dna_ambig_example_logl_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
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

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let c_w_x = SCB::new(model.clone(), info_w_x).build().unwrap();
    let c_w_n = SCB::new(model, info_w_n).build().unwrap();

    assert_relative_eq!(c_w_x.cost(), c_w_n.cost());
}

#[test]
fn dna_ambig_example_likelihood() {
    dna_ambig_example_logl_template::<JC69>(&[], &[]);
    dna_ambig_example_logl_template::<K80>(&[], &[]);
    dna_ambig_example_logl_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    dna_ambig_example_logl_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    dna_ambig_example_logl_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
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
    let k80 = SubstModel::<K80>::new(&[], &[2.0, 1.0]).unwrap();
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
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]).unwrap();
    let c_gtr = SCB::new(gtr_as_hky, info).build().unwrap();
    assert_relative_eq!(c_gtr.cost(), -216.234734, epsilon = 1e-3);
}

#[cfg(test)]
fn protein_example_logl_template<Q: QMatrix + QMatrixMaker>(
    params: &[f64],
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
    let model = SubstModel::<Q>::new(&[], params).unwrap();
    let c = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), expected_llik, epsilon = epsilon);
}

#[test]
fn protein_example_likelihood() {
    protein_example_logl_template::<WAG>(&[], -4505.736814460457, 1e-3);
    protein_example_logl_template::<HIVB>(&[], -4407.989226397638, 1e-5);
    protein_example_logl_template::<BLOSUM>(&[], -4587.71053, 1e-5);
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
fn logl_revers_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64], epsilon: f64) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let (info, info_rerooted) = simple_reroot_info(Q::alphabet());

    let c = SCB::new(model.clone(), info).build().unwrap();
    let c_rerooted = SCB::new(model, info_rerooted).build().unwrap();
    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = epsilon,);
}

#[test]
fn dna_logl_reversibility() {
    logl_revers_template::<JC69>(&[], &[], 1e-10);
    logl_revers_template::<K80>(&[], &[], 1e-10);
    logl_revers_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5], 1e-10);
    logl_revers_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
        1e-10,
    );
    logl_revers_template::<GTR>(
        &[0.22, 0.26, 0.33, 0.19],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
        1e-10,
    );
}

#[test]
fn protein_logl_reversibility() {
    logl_revers_template::<WAG>(&[], &[], 1e-8);
    logl_revers_template::<HIVB>(&[], &[], 1e-8);
    logl_revers_template::<BLOSUM>(&[], &[], 1e-3);
}

fn huelsenbeck_reversibility_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
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
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let c_rerooted = SCB::new(model, info_rerooted).build().unwrap();
    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = 1e-10,);
}

#[test]
fn huelsenbeck_reversibility() {
    huelsenbeck_reversibility_template::<JC69>(&[], &[]);
    huelsenbeck_reversibility_template::<K80>(&[], &[0.5]);
    huelsenbeck_reversibility_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    huelsenbeck_reversibility_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    huelsenbeck_reversibility_template::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    );
}

#[cfg(test)]
fn logl_correct_w_diff_info<Q: QMatrix + QMatrixMaker>(llik1: f64, llik2: f64) {
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

    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let c1 = SCB::new(model.clone(), info1).build().unwrap();
    let c2 = SCB::new(model, info2).build().unwrap();

    assert_relative_eq!(c1.cost(), llik1, epsilon = 1e-5);
    assert_relative_eq!(c2.cost(), llik2, epsilon = 1e-5);
}

#[test]
fn protein_logl_correct_w_diff_info() {
    logl_correct_w_diff_info::<WAG>(-7.488595394504073, -10.206456536551775);
    logl_correct_w_diff_info::<HIVB>(-7.231597482410509, -9.865559545434952);
    logl_correct_w_diff_info::<BLOSUM>(-7.4408154253528975, -10.33187874481282);
}

fn one_site_one_char_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    // This used to fail on leaf data creation when some of the sequences were empty
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
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
    one_site_one_char_template::<JC69>(&[], &[]);
    one_site_one_char_template::<K80>(&[], &[]);
    one_site_one_char_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    one_site_one_char_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    one_site_one_char_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn protein_one_site_one_char() {
    one_site_one_char_template::<WAG>(&[], &[]);
    one_site_one_char_template::<HIVB>(&[], &[]);
    one_site_one_char_template::<BLOSUM>(&[], &[]);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn hiv_subset_valid_subst_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PIB::new(alignment).build().unwrap();
    let gtr =
        SubstModel::<GTR>::new(&[0.25, 0.25, 0.25, 0.25], &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).unwrap();
    let c = SCB::new(gtr, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
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
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69, info).build().unwrap();

    // Compare against value from PhyML
    assert_relative_eq!(c.cost(), -9.70406054783923);
}

#[test]
fn dna_single_char_gaps_against_phyml() {
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
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
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
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
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
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
fn x_fully_likely_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
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
    x_fully_likely_template::<JC69>(&[], &[]);
    x_fully_likely_template::<K80>(&[], &[]);
    x_fully_likely_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    x_fully_likely_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    x_fully_likely_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0]);
}

#[test]
fn protein_x_fully_likely() {
    x_fully_likely_template::<WAG>(&[], &[]);
    x_fully_likely_template::<HIVB>(&[], &[]);
    x_fully_likely_template::<BLOSUM>(&[], &[]);
}

#[cfg(test)]
fn avg_rate_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let avg_rate = model.q().diagonal().component_mul(model.freqs()).sum();
    assert_relative_eq!(avg_rate, -1.0, epsilon = 1e-10);
}

#[test]
fn dna_avg_rate() {
    avg_rate_template::<JC69>(&[], &[]);
    avg_rate_template::<K80>(&[], &[]);
    avg_rate_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    avg_rate_template::<TN93>(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435, 0.00135]);
    avg_rate_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]);
}

#[test]
fn protein_avg_rate() {
    avg_rate_template::<WAG>(&[], &[]);
    avg_rate_template::<HIVB>(&[], &[]);
    avg_rate_template::<BLOSUM>(&[], &[]);
    let freqs = &[1.0 / 20.0; 20];
    avg_rate_template::<WAG>(freqs, &[]);
    avg_rate_template::<HIVB>(freqs, &[]);
    avg_rate_template::<BLOSUM>(freqs, &[]);
}

#[test]
fn p_matrix_limit() {
    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let time = 1e10f64;
    let p = model.p(time);
    let p2 = model.p(1e3f64);
    check_freq_convergence(p.clone(), model.freqs(), 1e-5);
    check_freq_convergence(p2.clone(), model.freqs(), 1e-5);
    assert_relative_eq!(p, p2, epsilon = 1e-5);
}

#[cfg(test)]
fn parsimony_rounding_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
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
    parsimony_rounding_template::<HIVB>(&[], &[]);
    parsimony_rounding_template::<WAG>(&[], &[]);
    parsimony_rounding_template::<BLOSUM>(&[], &[]);
}

#[test]
fn dna_rounding_scores() {
    parsimony_rounding_template::<JC69>(&[], &[]);
    parsimony_rounding_template::<K80>(&[], &[]);
    parsimony_rounding_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    parsimony_rounding_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    parsimony_rounding_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]);
}

#[cfg(test)]
fn parsimony_zero_diag_template<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) {
    let model = SubstModel::<Q>::new(freqs, params).unwrap();
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
    parsimony_zero_diag_template::<HIVB>(&[], &[]);
    parsimony_zero_diag_template::<WAG>(&[], &[]);
    parsimony_zero_diag_template::<BLOSUM>(&[], &[]);
}

#[test]
fn dna_zero_diag_scores() {
    parsimony_zero_diag_template::<JC69>(&[], &[]);
    parsimony_zero_diag_template::<K80>(&[], &[]);
    parsimony_zero_diag_template::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]);
    parsimony_zero_diag_template::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    parsimony_zero_diag_template::<GTR>(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]);
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
fn dirty_tree_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    use crate::likelihood::TreeSearchCost;

    let info = setup_test_info(Q::alphabet());

    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
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
    dirty_tree_costs_match_template::<JC69>();
    dirty_tree_costs_match_template::<K80>();
    dirty_tree_costs_match_template::<HKY>();
    dirty_tree_costs_match_template::<TN93>();
    dirty_tree_costs_match_template::<GTR>();

    dirty_tree_costs_match_template::<WAG>();
    dirty_tree_costs_match_template::<HIVB>();
    dirty_tree_costs_match_template::<BLOSUM>();
}

#[cfg(test)]
fn dirty_branch_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    use crate::likelihood::TreeSearchCost;

    let info = setup_test_info(Q::alphabet());

    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
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
    dirty_branch_costs_match_template::<JC69>();
    dirty_branch_costs_match_template::<K80>();
    dirty_branch_costs_match_template::<HKY>();
    dirty_branch_costs_match_template::<TN93>();
    dirty_branch_costs_match_template::<GTR>();

    dirty_branch_costs_match_template::<WAG>();
    dirty_branch_costs_match_template::<HIVB>();
    dirty_branch_costs_match_template::<BLOSUM>();
}

#[cfg(test)]
fn modify_model_params_costs_match_template<Q: QMatrix + QMatrixMaker>() {
    let info = setup_test_info(Q::alphabet());

    let model = SubstModel::<Q>::new(&[], &[1.0]).unwrap();
    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = c.cost();

    // The likelihood should change if we change model parameters
    c.set_param(0, 0.5);

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let new_model = SubstModel::<Q>::new(&[], &[0.5]).unwrap();
    let c = SCB::new(new_model, info).build().unwrap();
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_params_costs_match() {
    // does not apply to JC69, WAG, HIVB, BLOSUM which have no params
    modify_model_params_costs_match_template::<K80>();
    modify_model_params_costs_match_template::<HKY>();
    modify_model_params_costs_match_template::<TN93>();
    modify_model_params_costs_match_template::<GTR>();
}

#[cfg(test)]
fn modify_model_freqs_costs_match_template<Q: QMatrix + QMatrixMaker>(freqs: FreqVector) {
    let info = setup_test_info(Q::alphabet());

    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let mut c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let logl = c.cost();

    // The likelihood should change if we change model frequencies
    c.set_freqs(freqs.clone()).unwrap();

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification
    let new_model = SubstModel::<Q>::new(freqs.as_slice(), &[]).unwrap();
    let c = SCB::new(new_model, info).build().unwrap();
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_freqs_costs_match() {
    // does not apply to JC69 and K80 which have no freqs
    let new_dna_freqs = frequencies!(&[0.1, 0.1, 0.1, 0.7]);
    modify_model_freqs_costs_match_template::<HKY>(new_dna_freqs.clone());
    modify_model_freqs_costs_match_template::<TN93>(new_dna_freqs.clone());
    modify_model_freqs_costs_match_template::<GTR>(new_dna_freqs);

    let new_aa_freqs = frequencies!(&[0.05; 20]);
    modify_model_freqs_costs_match_template::<WAG>(new_aa_freqs.clone());
    modify_model_freqs_costs_match_template::<BLOSUM>(new_aa_freqs.clone());
    modify_model_freqs_costs_match_template::<HIVB>(new_aa_freqs);
}
