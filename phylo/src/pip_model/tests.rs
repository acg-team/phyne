use std::path::Path;

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector};

use crate::alignment::{Alignment, Sequences, MSA};
use crate::alphabets::{Alphabet, AMINOACIDS as aas, GAP, NUCLEOTIDES as nucls};
use crate::error::{Error, EvolutionaryModelError};
use crate::evolutionary_models::EvoModel;
use crate::io::read_sequences;
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCost, PIPCostBuilder as PIPB, PIPModel, PIPModelInfo, PIP_PARAM_N};
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, QMatrixMaker, SubstMatrix, SubstModel,
};
use crate::{frequencies, record_wo_desc as record, tree};

const UNNORMALIZED_PIP_HKY_Q: [f64; 25] = [
    -0.9, 0.11, 0.22, 0.22, 0.0, 0.13, -0.88, 0.26, 0.26, 0.0, 0.33, 0.33, -0.825, 0.165, 0.0,
    0.19, 0.19, 0.095, -0.895, 0.0, 0.25, 0.25, 0.25, 0.25, -0.0,
];

#[cfg(test)]
const EQUAL_DNA_FREQS: [f64; 4] = [0.25; 4];

#[cfg(test)]
fn compare_pip_subst_rates_template<Q: QMatrix + Default>(chars: &[u8]) {
    let pip_model = PIPModel::<Q>::with_default_substitution(&[0.1, 0.4]).unwrap();
    let subst_model = SubstModel::<Q>::default();
    for (i, &char) in chars.iter().enumerate() {
        assert!(pip_model.rate(char, char) < 0.0);
        assert_relative_eq!(pip_model.q.row(i).sum(), 0.0, epsilon = 1e-10);
        for &other_char in chars {
            if char == other_char {
                continue;
            }
            assert_relative_eq!(
                pip_model.rate(char, other_char),
                subst_model.rate(char, other_char),
                epsilon = 1e-10
            );
        }
        assert_relative_eq!(pip_model.rate(char, GAP), pip_model.mu());
        assert_relative_eq!(pip_model.rate(GAP, char), 0.0);
    }
}

#[test]
fn pip_dna_jc69_correct() {
    let pip_jc69 = PIPModel::<JC69>::with_default_substitution(&[0.1, 0.4]).unwrap();
    assert_eq!(pip_jc69.lambda(), 0.1);
    assert_eq!(pip_jc69.mu(), 0.4);
    assert_eq!(
        pip_jc69.freqs(),
        &frequencies!(&[0.25, 0.25, 0.25, 0.25, 0.0])
    );
    assert!(pip_jc69
        .q
        .diagonal()
        .iter()
        .take(pip_jc69.q.nrows() - 2)
        .all(|&x| x == -1.0 - 0.4));
}

#[test]
fn pip_dna_k80_correct() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let pip_k80 = PIPModel::<K80>::new(&EQUAL_DNA_FREQS, &[lambda, mu, kappa]).unwrap();
    assert_eq!(pip_k80.lambda(), lambda);
    assert_eq!(pip_k80.mu(), mu);
    assert_eq!(
        pip_k80.freqs(),
        &frequencies!(&[0.25, 0.25, 0.25, 0.25, 0.0])
    );
    assert_eq!(pip_k80.subst_q.params(), &[kappa]);
    assert!(pip_k80
        .q
        .diagonal()
        .iter()
        .take(pip_k80.q.nrows() - 2)
        .all(|&x| x == -1.0 - mu));
}

#[test]
fn pip_dna_hky_correct() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let freqs = &[0.22, 0.26, 0.33, 0.19];
    let pip_hky = PIPModel::<HKY>::new(freqs, &[lambda, mu, kappa]).unwrap();
    assert_eq!(pip_hky.lambda(), lambda);
    assert_eq!(pip_hky.mu(), mu);
    assert_eq!(pip_hky.freqs(), &frequencies!(freqs).insert_row(4, 0.0));
    assert_eq!(pip_hky.subst_q.params(), &[kappa]);
}

#[test]
fn pip_dna_hky_as_k80() {
    let lambda = 0.3;
    let mu = 0.7;
    let kappa = 0.5;
    let pip_k80 = PIPModel::<K80>::new(&EQUAL_DNA_FREQS, &[lambda, mu, kappa]).unwrap();
    let pip_hky = PIPModel::<HKY>::new(&EQUAL_DNA_FREQS, &[lambda, mu, kappa]).unwrap();
    assert_eq!(pip_k80.lambda(), pip_hky.lambda());
    assert_eq!(pip_k80.mu(), pip_hky.mu());
    assert_eq!(pip_k80.freqs(), pip_hky.freqs());
    assert_eq!(pip_k80.subst_q.params(), pip_hky.subst_q.params());
    assert!(pip_hky
        .q
        .diagonal()
        .iter()
        .take(pip_hky.q.nrows() - 2)
        .all(|&x| x == -1.0 - mu));
}

#[test]
fn pip_dna_subst_rates() {
    compare_pip_subst_rates_template::<JC69>(nucls);
    compare_pip_subst_rates_template::<K80>(nucls);
    compare_pip_subst_rates_template::<HKY>(nucls);
    compare_pip_subst_rates_template::<TN93>(nucls);
    compare_pip_subst_rates_template::<GTR>(nucls);
}

#[test]
fn pip_protein_subst_rates() {
    compare_pip_subst_rates_template::<WAG>(aas);
    compare_pip_subst_rates_template::<HIVB>(aas);
    compare_pip_subst_rates_template::<BLOSUM>(aas);
}

#[cfg(test)]
fn pip_normalised_check_template<Q: QMatrix>(pip: PIPModel<Q>, chars: &[u8]) {
    assert_eq!(pip.lambda(), pip.params()[0]);
    assert_eq!(pip.mu(), pip.params()[1]);
    let stat_dist = pip.freqs().clone();
    assert_relative_eq!(pip.freqs(), &stat_dist);
    assert_relative_eq!(pip.q.sum(), 0.0, epsilon = 1e-10);
    for &char in chars {
        let mut sum = 0.0;
        for &other_char in chars {
            sum += pip.rate(char, other_char);
        }
        assert_relative_eq!(sum, 0.0, epsilon = 1e-14);
        assert_relative_eq!(pip.rate(char, GAP), pip.params()[1]);
        assert_relative_eq!(pip.rate(GAP, char), 0.0);
    }
}

#[test]
fn pip_dna_normalised() {
    pip_normalised_check_template(
        PIPModel::<JC69>::with_default_substitution(&[0.2, 0.5]).unwrap(),
        nucls,
    );
    pip_normalised_check_template(
        PIPModel::<K80>::with_default_substitution(&[0.1, 1.5]).unwrap(),
        nucls,
    );
    pip_normalised_check_template(
        PIPModel::<HKY>::with_default_substitution(&[0.05, 0.7]).unwrap(),
        nucls,
    );
    pip_normalised_check_template(
        PIPModel::<TN93>::with_default_substitution(&[0.05, 0.7]).unwrap(),
        nucls,
    );
    pip_normalised_check_template(
        PIPModel::<GTR>::with_default_substitution(&[0.05, 0.7]).unwrap(),
        nucls,
    );
}

#[test]
fn pip_protein_normalised() {
    pip_normalised_check_template(
        PIPModel::<WAG>::with_default_substitution(&[0.2, 0.5]).unwrap(),
        aas,
    );
    pip_normalised_check_template(
        PIPModel::<HIVB>::with_default_substitution(&[0.1, 1.5]).unwrap(),
        aas,
    );
    pip_normalised_check_template(
        PIPModel::<BLOSUM>::with_default_substitution(&[0.05, 0.7]).unwrap(),
        aas,
    );
}

#[cfg(test)]
fn pip_infinity_p_template<Q: QMatrix>(model: PIPModel<Q>) {
    let p_inf = model.p(10000.0);
    assert_eq!(p_inf.shape(), model.q().shape());
    for row in p_inf.row_iter() {
        assert_relative_eq!(row.sum(), 1.0, epsilon = 1e-10);
        assert_relative_eq!(row.sum() - row[row.len() - 1], 0.0, epsilon = 1e-10);
        assert_relative_eq!(row[row.len() - 1], 1.0, epsilon = 1e-10);
    }
}

#[test]
fn pip_dna_infinity_p() {
    pip_infinity_p_template(PIPModel::<JC69>::with_default_substitution(&[0.2, 0.5]).unwrap());
    pip_infinity_p_template(PIPModel::<K80>::with_default_substitution(&[0.1, 1.5]).unwrap());
    pip_infinity_p_template(
        PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.2, 0.3, 0.5]).unwrap(),
    );
    pip_infinity_p_template(
        PIPModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.1, 0.4, 0.5970915, 0.2940435])
            .unwrap(),
    );
    pip_infinity_p_template(
        PIPModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[0.2, 0.5, 5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[test]
fn pip_protein_infinity_p() {
    pip_infinity_p_template(PIPModel::<WAG>::with_default_substitution(&[0.2, 0.5]).unwrap());
    pip_infinity_p_template(PIPModel::<HIVB>::with_default_substitution(&[0.2, 0.5]).unwrap());
    pip_infinity_p_template(PIPModel::<BLOSUM>::with_default_substitution(&[0.2, 0.3]).unwrap());
}

#[test]
fn pip_dna_tn93_correct() {
    let pip_tn93 =
        PIPModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.2, 0.5, 0.5970915, 0.2940435])
            .unwrap();
    let tn93 = SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap();
    let mut diff = SubstMatrix::zeros(4, 4);
    diff.fill_diagonal(-0.5);
    diff = diff.insert_column(4, 0.5).insert_row(4, 0.0);
    let expected_q = tn93.q().clone().insert_column(4, 0.0).insert_row(4, 0.0) + diff;
    assert_relative_eq!(pip_tn93.q, expected_q, epsilon = 1e-10);
    assert_relative_eq!(
        pip_tn93.freqs(),
        &frequencies!(&[0.22, 0.26, 0.33, 0.19, 0.0])
    );
}

#[test]
fn pip_p_example_matrix() {
    // PIP matrix example from the PIP likelihood tutorial, rounded to 3 decimal values
    let epsilon = 1e-3;
    let mut pip_hky = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    pip_hky.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.225, 0.109, 0.173, 0.0996, 0.393, 0.0922, 0.242, 0.173, 0.0996, 0.393, 0.115, 0.136,
            0.276, 0.0792, 0.393, 0.115, 0.136, 0.138, 0.217, 0.393, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.p(2.0), expected_p, epsilon = epsilon);
    let expected_p = SubstMatrix::from_row_slice(
        5,
        5,
        &[
            0.437, 0.0859, 0.162, 0.0935, 0.221, 0.0727, 0.45, 0.162, 0.0935, 0.221, 0.108, 0.128,
            0.48, 0.0625, 0.221, 0.108, 0.128, 0.108, 0.434, 0.221, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    );
    assert_relative_eq!(pip_hky.p(1.0), expected_p, epsilon = epsilon);
}

#[cfg(test)]
fn setup_example_phylo_info() -> PhyloInfo<MSA> {
    let tree = tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"-A--"),
            record!("B", b"CA--"),
            record!("C", b"-A-G"),
            record!("D", b"-CAA"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn assert_values<Q: QMatrix>(
    tmp: &PIPModelInfo<Q>,
    idx: usize,
    exp_survins: f64,
    exp_anc: &[f64],
    exp_f: &[f64],
    exp_pnu: &[f64],
) {
    let e = 1e-3;
    assert_relative_eq!(tmp.surv_ins_weights[idx], exp_survins, epsilon = e);
    assert_relative_eq!(
        tmp.f[idx],
        DVector::<f64>::from_column_slice(exp_f),
        epsilon = e
    );
    assert_eq!(tmp.anc[idx].nrows(), exp_f.len());
    assert_eq!(tmp.anc[idx].ncols(), 3);
    assert_relative_eq!(
        tmp.anc[idx].as_slice(),
        DMatrix::<f64>::from_column_slice(exp_anc.len(), 1, exp_anc).as_slice(),
    );
    assert_relative_eq!(
        tmp.pnu[idx],
        DVector::<f64>::from_column_slice(exp_pnu),
        epsilon = e
    );
}

#[test]
fn pip_hky_likelihood_example_leaf_values() {
    let info = setup_example_phylo_info();
    let tree = info.tree.clone();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let nu = 7.5;
    let ib = 0.1333 * 0.78694; // iota times beta

    let c = PIPB::new(model, info).build().unwrap();
    c.cost();
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("A")),
        nu * ib,
        &[[0.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.0],
        &[0.0, 0.33 * nu * ib, 0.0, 0.0],
    );
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("B")),
        nu * ib,
        &[[1.0, 1.0, 0.0, 0.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.26, 0.33, 0.0, 0.0],
        &[0.26 * nu * ib, 0.33 * nu * ib, 0.0, 0.0],
    );

    let ib = 0.0667 * 0.8848; // iota times beta
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("C")),
        nu * ib,
        &[[0.0, 1.0, 0.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.33, 0.0, 0.19],
        &[0.0, 0.33 * nu * ib, 0.0, 0.19 * nu * ib],
    );
    assert_values(
        &c.tmp.borrow(),
        usize::from(tree.idx("D")),
        nu * ib,
        &[[0.0, 1.0, 1.0, 1.0], [0.0; 4], [0.0; 4]].concat(),
        &[0.0, 0.26, 0.33, 0.33],
        &[0.0, 0.26 * nu * ib, 0.33 * nu * ib, 0.33 * nu * ib],
    );
}

#[test]
fn pip_hky_likelihood_example_internals() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    let ib_e = 0.1333 * 0.78694; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("E")),
        nu * ib_e,
        &[1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        &[0.0619, 0.0431, 0.154, 0.154],
        &[nu * ib_e * (0.0619 + 0.26), nu * ib_e * (0.0431), 0.0, 0.0],
    );

    let ib_f = 0.2 * 0.70351; // iota times beta
    let ib_d = 0.0667 * 0.8848; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("F")),
        nu * ib_f,
        &[0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        &[0.0488, 0.0449, 0.0567, 0.0261],
        &[
            0.0,
            nu * (0.0449 * ib_f),
            nu * (0.0567 * ib_f + 0.33 * ib_d),
            nu * (0.0261 * ib_f),
        ],
    );

    let ib_r = 0.2667 * 1.0; // iota times beta
    assert_values(
        &tmp,
        usize::from(info.tree.idx("R")),
        nu * ib_r,
        &[1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
        &[0.0207, 0.000557, 0.013, 0.00598],
        &[
            nu * (0.0207 * ib_r + (0.0619 + 0.26) * ib_e),
            nu * (0.000557 * ib_r),
            nu * (0.013 * ib_r + 0.0567 * ib_f + 0.33 * ib_d),
            nu * (0.00598 * ib_r + 0.0261 * ib_f),
        ],
    );
}

#[cfg(test)]
fn assert_c0_values<Q: QMatrix>(tmp: &PIPModelInfo<Q>, idx: usize, exp_f1: f64, exp_pnu: f64) {
    let e = 1e-3;
    assert_relative_eq!(tmp.c0_f1[idx], exp_f1, epsilon = e);
    assert_relative_eq!(tmp.c0_pnu[idx], exp_pnu, epsilon = e);
}

#[test]
fn pip_hky_likelihood_example_c0() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("A")),
        -1.0,
        nu * 0.028408 - 1.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("B")),
        -1.0,
        nu * 0.028408 - 1.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("C")),
        -1.0,
        nu * 0.00768 - 0.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("D")),
        -1.0,
        nu * 0.00768 - 0.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("E")),
        -0.84518,
        nu * (0.044652 + 0.028408 * 2.0) - 3.0,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("F")),
        -0.9512,
        nu * 0.066164 + nu * 0.00768 * 2.0 - 2.5,
    );
    assert_c0_values(
        &tmp,
        usize::from(info.tree.idx("R")),
        -0.732,
        nu * (0.071467 + 0.044652 + 0.028408 * 2.0 + 0.066164 + 0.00768 * 2.0 - 1.0),
    );
}

#[test]
fn pip_hky_likelihood_example_final() {
    let info = setup_example_phylo_info();
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);

    let c = PIPB::new(model, info.clone()).build().unwrap();
    c.cost();
    let tmp = c.tmp.borrow();

    let nu = 7.5;
    assert_relative_eq!(
        tmp.pnu[0],
        DVector::from_column_slice(&[
            nu * 0.0392204949,
            nu * 0.000148719,
            nu * 0.03102171,
            nu * 0.00527154
        ]),
        epsilon = 1e-3
    );
    assert_relative_eq!(tmp.c0_pnu[0], -5.591, epsilon = 1e-3);
    assert_relative_eq!(
        c.cost(),
        -20.769363665853653 - 0.709020450847471,
        epsilon = 1e-3
    );
    assert_relative_eq!(c.cost(), -21.476307347643274, epsilon = 1e-2); // value from the python script
}

#[cfg(test)]
fn setup_example_phylo_info_2() -> PhyloInfo<MSA> {
    let tree = tree!("((A:2,B:2)E:2,(C:1,D:1)F:3)R:0;");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"--A--"),
            record!("B", b"-CA--"),
            record!("C", b"--A-G"),
            record!("D", b"T-CAA"),
        ]),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[test]
fn pip_hky_likelihood_example_2() {
    let mut model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    model.q = SubstMatrix::from_column_slice(5, 5, &UNNORMALIZED_PIP_HKY_Q);
    let info = setup_example_phylo_info_2();
    let c = PIPB::new(model, info).build().unwrap();
    assert_relative_eq!(c.cost(), -24.9549393298, epsilon = 1e-2);
}

#[test]
fn pip_likelihood_huelsenbeck_example_hky() {
    let info = PIB::with_attrs(
        "./data/Huelsenbeck_example_long_DNA.fasta",
        "./data/Huelsenbeck_example.newick",
    )
    .build()
    .unwrap();
    let model = PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 0.25, 0.5]).unwrap();
    let mut c = PIPB::new(model, info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -372.1419048976677, epsilon = 1e-6);

    // Check that model update works
    c.set_param(0, 1.2);
    c.set_param(1, 0.45);
    c.set_freqs(frequencies!(&[0.25, 0.25, 0.25, 0.25]))
        .unwrap();
    c.set_param(2, 1.0);
    assert_relative_eq!(c.cost(), -361.18634412281443, epsilon = 1e-6); // value from the python script
}

#[test]
fn pip_likelihood_huelsenbeck_example_gtr() {
    let info = PIB::with_attrs(
        "./data/Huelsenbeck_example_long_DNA.fasta",
        "./data/Huelsenbeck_example.newick",
    )
    .build()
    .unwrap();
    let model = PIPModel::<GTR>::new(
        &[0.22, 0.26, 0.33, 0.19],
        &[
            0.5,
            0.25,
            0.8532418333548707,
            0.7308730811869607,
            0.6801286803463216,
            0.7800327822023927,
            1.042256394909917,
        ],
    )
    .unwrap();
    let c = PIPB::new(model, info).build().unwrap();

    assert_relative_eq!(c.cost(), -359.2342943608156, epsilon = 1e-4);
}

#[test]
fn pip_likelihood_huelsenbeck_example_model_comp() {
    let info = PIB::with_attrs(
        "./data/Huelsenbeck_example_long_DNA.fasta",
        "./data/Huelsenbeck_example.newick",
    )
    .build()
    .unwrap();

    let jc69 = PIPModel::<JC69>::new(&EQUAL_DNA_FREQS, &[1.1, 0.55]).unwrap();
    let c = PIPB::new(jc69, info.clone()).build().unwrap();

    let hky_as_jc = PIPModel::<HKY>::new(&EQUAL_DNA_FREQS, &[1.1, 0.55, 1.0]).unwrap();
    let c_hky = PIPB::new(hky_as_jc, info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), c_hky.cost());
}

#[test]
fn pip_likelihood_huelsenbeck_example_reroot() {
    let phylo = PIB::with_attrs(
        "./data/Huelsenbeck_example_long_DNA.fasta",
        "./data/Huelsenbeck_example.newick",
    )
    .build()
    .unwrap();
    let model_gtr = PIPModel::<GTR>::new(
        &[0.22, 0.26, 0.33, 0.19],
        &[
            0.5,
            0.25,
            0.8532418333548707,
            0.7308730811869607,
            0.6801286803463216,
            0.7800327822023927,
            1.042256394909917,
        ],
    )
    .unwrap();
    let phylo_rerooted = PIB::with_attrs(
        "./data/Huelsenbeck_example_long_DNA.fasta",
        "./data/Huelsenbeck_example_reroot.newick",
    )
    .build()
    .unwrap();
    let c = PIPB::new(model_gtr.clone(), phylo).build().unwrap();
    let c_rerooted = PIPB::new(model_gtr, phylo_rerooted).build().unwrap();

    assert_relative_eq!(c.cost(), c_rerooted.cost(), epsilon = 1e-6);
    assert_relative_eq!(c.cost(), -359.2342943608156, epsilon = 1e-6);
}

#[test]
fn pip_likelihood_protein_example() {
    let info = PIB::with_attrs(
        "./data/phyml_protein_example/seqs.fasta",
        "./data/phyml_protein_example/example_tree.newick",
    )
    .build()
    .unwrap();

    let model_wag = PIPModel::<WAG>::new(WAG_PI.as_slice(), &[0.5, 0.25]).unwrap();
    let c = PIPB::new(model_wag, info.clone()).build().unwrap();
    assert!(c.cost() <= 0.0);
    // The cost is the same when initialising the model with default frequencies
    assert_eq!(
        PIPB::new(
            PIPModel::<WAG>::with_default_substitution(&[0.5, 0.25]).unwrap(),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );

    // The cost is different when initialising the model with equal frequencies
    assert_ne!(
        PIPB::new(
            PIPModel::<WAG>::new(&[0.05; 20], &[0.5, 0.25]).unwrap(),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );

    // The cost is different when initialising the model with default frequencies but different mu/lambda
    assert_ne!(
        PIPB::new(
            PIPModel::<WAG>::with_default_substitution(&[0.5, 0.2]).unwrap(),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );
    assert_ne!(
        PIPB::new(
            PIPModel::<WAG>::with_default_substitution(&[0.1, 0.25]).unwrap(),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );
    assert_ne!(
        PIPB::new(
            PIPModel::<WAG>::with_default_substitution(&[0.1, 0.2]).unwrap(),
            info.clone()
        )
        .build()
        .unwrap()
        .cost(),
        c.cost(),
    );
}

#[test]
fn designation() {
    let model = PIPModel::<JC69>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("lambda = 1.5"));
    assert!(format!("{model}").contains("mu = 1.5"));
    assert!(format!("{model}").contains("JC69"));

    let model = PIPModel::<JC69>::new(&[0.25; 4], &[2.0, 1.0]).unwrap();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("lambda = 2.0"));
    assert!(format!("{model}").contains("mu = 1.0"));
    assert!(format!("{model}").contains("JC69"));

    let model = PIPModel::<K80>::new(&[0.25; 4], &[2.0, 5.0, 1.3]).unwrap();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("lambda = 2.0"));
    assert!(format!("{model}").contains("mu = 5.0"));
    assert!(format!("{model}").contains("K80"));
    assert!(format!("{model}").contains("kappa = 1.3"));

    let model = PIPModel::<HKY>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("lambda = 1.5"));
    assert!(format!("{model}").contains("mu = 1.5"));
    assert!(format!("{model}").contains("HKY"));
    assert!(format!("{model}").contains("kappa = 2.0"));

    let model = PIPModel::<TN93>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("lambda = 1.5"));
    assert!(format!("{model}").contains("mu = 1.5"));
    assert!(format!("{model}").contains("TN93"));

    let model = PIPModel::<GTR>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("GTR"));

    let model = PIPModel::<WAG>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("WAG"));

    let model = PIPModel::<HIVB>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("HIVB"));

    let model = PIPModel::<BLOSUM>::default();
    assert!(format!("{model}").contains("PIP"));
    assert!(format!("{model}").contains("BLOSUM"));
}

#[test]
fn pip_logl_correct_w_diff_info() {
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

    let pip_wag = PIPModel::<WAG>::with_default_substitution(&[50.0, 0.1]).unwrap();

    let c1 = PIPB::new(pip_wag.clone(), info1).build().unwrap();
    let c2 = PIPB::new(pip_wag, info2).build().unwrap();

    assert_relative_eq!(c1.cost(), -854.2260753055998, epsilon = 1e-2);
    assert_relative_eq!(c2.cost(), -1125.1290016747846, epsilon = 1e-2);
    assert_ne!(c1.cost(), c2.cost());
}

#[cfg(test)]
fn avg_rate_template<Q: QMatrix>(model: PIPModel<Q>) {
    let mu = model.mu();
    let n = model.q().nrows();
    let avg_rate = model
        .q()
        .view((0, 0), (n - 1, n - 1))
        .diagonal()
        .component_mul(&model.freqs().view((0, 0), (n - 1, 1)))
        .sum();
    assert_relative_eq!(avg_rate, -1.0 - mu, epsilon = 1e-6);
}

#[test]
fn dna_avg_rate() {
    avg_rate_template(PIPModel::<JC69>::new(&EQUAL_DNA_FREQS, &[0.24, 1.4]).unwrap());
    avg_rate_template(PIPModel::<K80>::new(&EQUAL_DNA_FREQS, &[0.4, 4.4, 0.5]).unwrap());
    avg_rate_template(PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5, 1.5, 0.5]).unwrap());
    avg_rate_template(
        PIPModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[1.5, 0.25, 0.5, 0.2]).unwrap(),
    );
    avg_rate_template(
        PIPModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 5.0, 5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    );
}

#[test]
fn protein_avg_rate() {
    avg_rate_template(PIPModel::<WAG>::with_default_substitution(&[0.5, 1.0]).unwrap());
    avg_rate_template(PIPModel::<HIVB>::with_default_substitution(&[5.0, 1.5]).unwrap());
    avg_rate_template(PIPModel::<BLOSUM>::with_default_substitution(&[4.5, 1.3]).unwrap());
    let freqs = &[1.0 / 20.0; 20];
    avg_rate_template(PIPModel::<WAG>::new(freqs, &[0.5, 1.0]).unwrap());
    avg_rate_template(PIPModel::<HIVB>::new(freqs, &[2.5, 0.03]).unwrap());
    avg_rate_template(PIPModel::<BLOSUM>::new(freqs, &[0.25, 1.5]).unwrap());
}

#[test]
fn logl_not_inf_for_empty_col() {
    let tree = tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(read_sequences("./data/sequences_empty_col.fasta").unwrap()),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };
    let model = PIPModel::<WAG>::with_default_substitution(&[0.5, 0.5]).unwrap();
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn blen_leading_to_small_probs() {
    let fldr = Path::new("./data/");
    let seq_file = fldr.join("p105.msa.fa");
    let tree_file = fldr.join("p105.newick");
    let info = PIB::with_attrs(seq_file, tree_file).build().unwrap();

    let model = PIPModel::<WAG>::default();
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}

#[test]
fn blen_leading_to_minusinf() {
    let tree = tree!("((284811:0.0000000000000002,(284593:0.1,(237561:0.3,(284812:0.3,(284813:400.9,284591:0.2):40000000000000.2):0.05):0.1):0.04):0);");
    let msa = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![
                record!("284813", b"-"),
                record!("284811", b"W"),
                record!("284593", b"W"),
                record!("237561", b"W"),
                record!("284591", b"W"),
                record!("284812", b"W"),
            ],
            Alphabet::protein(),
        ),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };

    let model = PIPModel::<WAG>::default();
    let c = PIPB::new(model, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl.is_sign_negative());
}

#[cfg(test)]
fn setup_test_info(alphabet: &'static Alphabet) -> PhyloInfo<MSA> {
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![
                record!("A", b"CT-ATA-TA-TAC"),
                record!("B", b"ATATA--TATA-A"),
                record!("C", b"TTA--TATATA-T"),
                record!("D", b"TTA--TATATA-T"),
            ],
            alphabet,
        ),
        &tree,
    )
    .unwrap();
    PhyloInfo { msa, tree }
}

#[cfg(test)]
fn setup_test_pip_cost<Q: QMatrix>(model: PIPModel<Q>) -> PIPCost<Q, MSA> {
    let info = setup_test_info(Q::alphabet());
    PIPB::new(model, info).build().unwrap()
}

#[cfg(test)]
fn dirty_tree_costs_match_template<Q: QMatrix + QMatrixMaker + Default>(model: PIPModel<Q>) {
    use crate::likelihood::TreeSearchCost;

    let mut c = setup_test_pip_cost::<Q>(model.clone());
    let logl = TreeSearchCost::cost(&c);
    assert_eq!(logl, TreeSearchCost::cost(&c));

    // The likelihood should be the same if we make the tree dirty without changing it
    let mut tree = c.info.tree.clone();
    tree.dirty();
    c.update_tree(tree);
    assert_eq!(logl, TreeSearchCost::cost(&c));

    // The likelihood should be the same if we rebuild from scratch
    let model = PIPModel::<Q>::new(model.subst_q.freqs().as_slice(), model.params()).unwrap();
    let c2 = setup_test_pip_cost(model);
    let logl2 = TreeSearchCost::cost(&c2);
    assert_eq!(logl2, TreeSearchCost::cost(&c2));
    assert_eq!(logl, logl2);
}

#[test]
fn dirty_tree_costs_match() {
    dirty_tree_costs_match_template(PIPModel::<JC69>::default());
    dirty_tree_costs_match_template(PIPModel::<K80>::default());
    dirty_tree_costs_match_template(PIPModel::<HKY>::default());
    dirty_tree_costs_match_template(PIPModel::<TN93>::default());
    dirty_tree_costs_match_template(PIPModel::<GTR>::default());

    dirty_tree_costs_match_template(PIPModel::<WAG>::default());
    dirty_tree_costs_match_template(PIPModel::<HIVB>::default());
    dirty_tree_costs_match_template(PIPModel::<BLOSUM>::default());
}

#[cfg(test)]
fn dirty_branch_costs_match_template<Q: QMatrix + Default>() {
    use crate::likelihood::TreeSearchCost;

    let info = setup_test_info(Q::alphabet());

    let model = PIPModel::<Q>::default();
    let mut c = PIPB::new(model.clone(), info.clone()).build().unwrap();
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
    let c = PIPB::new(model, new_info).build().unwrap();
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
fn modify_model_params_costs_match_template<Q: QMatrix + QMatrixMaker + Default>(
    model: PIPModel<Q>,
) {
    let mut c = setup_test_pip_cost(model.clone());
    let logl = c.cost();

    // The likelihood should change if we change model parameters
    c.set_param(0, 0.5);

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification

    let new_params = &[0.5]
        .into_iter()
        .chain(model.params()[1..].iter().cloned())
        .collect::<Vec<_>>();
    let new_model = PIPModel::<Q>::new(model.subst_q.freqs().as_slice(), new_params).unwrap();
    let c = setup_test_pip_cost(new_model);
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_params_costs_match() {
    modify_model_params_costs_match_template(PIPModel::<JC69>::default());
    modify_model_params_costs_match_template(PIPModel::<K80>::default());
    modify_model_params_costs_match_template(PIPModel::<HKY>::default());
    modify_model_params_costs_match_template(PIPModel::<TN93>::default());
    modify_model_params_costs_match_template(PIPModel::<GTR>::default());
}

#[cfg(test)]
fn modify_model_freqs_costs_match_template<Q: QMatrix + QMatrixMaker + Default>(
    model: PIPModel<Q>,
    freqs: &[f64],
) {
    let mut c = setup_test_pip_cost(model.clone());
    let logl = c.cost();

    // The likelihood should change if we change model frequencies
    c.set_freqs(frequencies!(freqs)).unwrap();

    let logl2 = c.cost();
    assert_eq!(logl2, c.cost());
    assert_ne!(logl, logl2);

    // The likelihood should be the same if we rebuild from scratch with the same modification

    let new_model = PIPModel::<Q>::new(freqs, model.params()).unwrap();
    let c = setup_test_pip_cost(new_model);
    let new_logl = c.cost();
    assert_eq!(new_logl, c.cost());
    assert_eq!(logl2, new_logl);
}

#[test]
fn modify_model_freqs_costs_match() {
    // does not apply to JC69 and K80 which have no freqs
    let new_dna_freqs = &[0.1, 0.1, 0.1, 0.7];
    modify_model_freqs_costs_match_template(PIPModel::<HKY>::default(), new_dna_freqs);
    modify_model_freqs_costs_match_template(PIPModel::<TN93>::default(), new_dna_freqs);
    modify_model_freqs_costs_match_template(PIPModel::<GTR>::default(), new_dna_freqs);

    let new_aa_freqs = &[0.05; 20];
    modify_model_freqs_costs_match_template(PIPModel::<WAG>::default(), new_aa_freqs);
    modify_model_freqs_costs_match_template(PIPModel::<HIVB>::default(), new_aa_freqs);
    modify_model_freqs_costs_match_template(PIPModel::<BLOSUM>::default(), new_aa_freqs);
}

#[cfg(test)]
fn incorrect_model_freqs_costs_match_template<Q: QMatrix + QMatrixMaker + Default>(
    model: PIPModel<Q>,
    freqs: &[f64],
) {
    let mut c = setup_test_pip_cost(model.clone());
    let logl = c.cost();

    assert!(c.set_freqs(frequencies!(freqs)).is_err());
    assert_eq!(logl, c.cost());
}

#[test]
fn incorrect_model_frequencies() {
    let new_dna_freqs = &[0.1, 0.1, 0.1, 0.6, 0.1];
    incorrect_model_freqs_costs_match_template(PIPModel::<JC69>::default(), new_dna_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<K80>::default(), new_dna_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<HKY>::default(), new_dna_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<TN93>::default(), new_dna_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<GTR>::default(), new_dna_freqs);

    let new_aa_freqs = &[1.0 / 30.0; 30];
    incorrect_model_freqs_costs_match_template(PIPModel::<WAG>::default(), new_aa_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<HIVB>::default(), new_aa_freqs);
    incorrect_model_freqs_costs_match_template(PIPModel::<BLOSUM>::default(), new_aa_freqs);
}

#[cfg(test)]
fn modify_model_wo_freqs_costs_match_template<Q: QMatrix + Default>(
    model: PIPModel<Q>,
    freqs: &[f64],
) {
    let mut c = setup_test_pip_cost(model);
    let logl = c.cost();

    assert!(c.set_freqs(frequencies!(freqs)).is_err());
    assert_eq!(logl, c.cost());
}

#[test]
fn modify_freqs_of_model_wo_freqs_fails() {
    let new_dna_freqs = &[0.1, 0.1, 0.1, 0.7];
    modify_model_wo_freqs_costs_match_template::<JC69>(PIPModel::<JC69>::default(), new_dna_freqs);
    modify_model_wo_freqs_costs_match_template::<K80>(PIPModel::<K80>::default(), new_dna_freqs);
}

#[cfg(test)]
fn pip_default_subst_template<Q: QMatrix + Default>() {
    let model = PIPModel::<Q>::with_default_substitution(&[1.5, 1.5]).unwrap();

    assert_eq!(
        &model.freqs().as_slice()[..model.n() - 1],
        model.subst_q.freqs().as_slice()
    );
    assert_eq!(model.params().len(), PIPModel::<Q>::param_count());
    assert_eq!(model.subst_q.params().len(), Q::param_count());
    assert_eq!(PIPModel::<Q>::param_count(), PIP_PARAM_N + Q::param_count());
}

#[test]
fn pip_default_subst_dna() {
    pip_default_subst_template::<JC69>();
    pip_default_subst_template::<K80>();
    pip_default_subst_template::<HKY>();
    pip_default_subst_template::<TN93>();
    pip_default_subst_template::<GTR>();
}

#[test]
fn pip_default_subst_protein() {
    pip_default_subst_template::<WAG>();
    pip_default_subst_template::<HIVB>();
    pip_default_subst_template::<BLOSUM>();
}

#[cfg(test)]
fn pip_default_subst_too_many_params_template<Q: QMatrix + Default>(params: &[f64]) {
    match PIPModel::<Q>::with_default_substitution(params) {
        Err(Error::EvolutionaryModel(EvolutionaryModelError::ParameterCount {
            name,
            actual,
            expected,
        })) => {
            assert_eq!(name, "PIP");
            assert_eq!(actual, params.len());
            assert_eq!(expected, PIP_PARAM_N);
        }
        _ => panic!(
            "Expected ParameterCount error for PIP with actual = {} and expected = {}",
            params.len(),
            PIP_PARAM_N
        ),
    }
}

#[test]
fn pip_default_subst_too_many_params_dna() {
    pip_default_subst_too_many_params_template::<JC69>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<K80>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<HKY>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<TN93>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<GTR>(&[0.5; PIP_PARAM_N + 1]);
}

#[test]
fn pip_default_subst_too_many_params_protein() {
    pip_default_subst_too_many_params_template::<WAG>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<HIVB>(&[0.5; PIP_PARAM_N + 1]);
    pip_default_subst_too_many_params_template::<BLOSUM>(&[0.5; PIP_PARAM_N + 1]);
}

#[cfg(test)]
fn pip_too_few_params_template<Q: QMatrix + QMatrixMaker>(params: &[f64]) {
    match PIPModel::<Q>::new(
        &vec![1.0 / Q::alphabet().len() as f64; Q::alphabet().len()],
        params,
    ) {
        Err(Error::EvolutionaryModel(EvolutionaryModelError::ParameterCount {
            name,
            actual,
            expected,
        })) => {
            assert_eq!(name, "PIP");
            assert_eq!(actual, params.len());
            assert_eq!(expected, PIPModel::<Q>::param_count());
        }
        _ => panic!(
            "Expected ParameterCount error for PIP with actual = {} and expected = {}",
            params.len(),
            PIPModel::<Q>::param_count()
        ),
    }
}

#[test]
fn pip_too_few_params_dna() {
    pip_too_few_params_template::<JC69>(&vec![0.5; JC69::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<K80>(&vec![0.5; K80::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<HKY>(&vec![0.5; HKY::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<TN93>(&vec![0.5; TN93::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<GTR>(&vec![0.5; GTR::param_count() + PIP_PARAM_N - 1]);
}

#[test]
fn pip_too_few_params_protein() {
    pip_too_few_params_template::<WAG>(&vec![0.5; WAG::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<HIVB>(&vec![0.5; HIVB::param_count() + PIP_PARAM_N - 1]);
    pip_too_few_params_template::<BLOSUM>(&vec![0.5; BLOSUM::param_count() + PIP_PARAM_N - 1]);
}

#[cfg(test)]
fn pip_default_subst_too_few_params_template<Q: QMatrix + Default>(params: &[f64]) {
    match PIPModel::<Q>::with_default_substitution(params) {
        Err(Error::EvolutionaryModel(EvolutionaryModelError::ParameterCount {
            name,
            actual,
            expected,
        })) => {
            assert_eq!(name, "PIP");
            assert_eq!(actual, params.len());
            assert_eq!(expected, PIP_PARAM_N);
        }
        _ => panic!(
            "Expected ParameterCount error for PIP with actual = {} and expected = {}",
            params.len(),
            PIP_PARAM_N
        ),
    }
}

#[test]
fn pip_default_subst_too_few_params_dna() {
    pip_default_subst_too_few_params_template::<JC69>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<K80>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<HKY>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<TN93>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<GTR>(&[0.5; 1]);
}

#[test]
fn pip_default_subst_too_few_params_protein() {
    pip_default_subst_too_few_params_template::<WAG>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<HIVB>(&[0.5; 1]);
    pip_default_subst_too_few_params_template::<BLOSUM>(&[0.5; 1]);
}

#[cfg(test)]
fn correct_parameter_number<Q: QMatrix + QMatrixMaker + Default>(model: PIPModel<Q>) {
    assert_eq!(model.params().len(), PIPModel::<Q>::param_count());
    assert_eq!(model.subst_q.params().len(), Q::param_count());
    assert_eq!(model.params().len(), Q::param_count() + PIP_PARAM_N);
}

#[test]
fn pip_correct_parameter_number_dna() {
    correct_parameter_number(
        PIPModel::<JC69>::new(&[0.25; 4], &vec![0.5; JC69::param_count() + PIP_PARAM_N]).unwrap(),
    );
    correct_parameter_number(
        PIPModel::<K80>::new(&[0.25; 4], &vec![0.5; K80::param_count() + PIP_PARAM_N]).unwrap(),
    );
    correct_parameter_number(
        PIPModel::<HKY>::new(&[0.25; 4], &vec![0.5; HKY::param_count() + PIP_PARAM_N]).unwrap(),
    );
    correct_parameter_number(
        PIPModel::<TN93>::new(&[0.25; 4], &vec![0.5; TN93::param_count() + PIP_PARAM_N]).unwrap(),
    );
    correct_parameter_number(
        PIPModel::<GTR>::new(&[0.25; 4], &vec![0.5; GTR::param_count() + PIP_PARAM_N]).unwrap(),
    );
}

#[test]
fn pip_correct_parameter_number_protein() {
    correct_parameter_number(PIPModel::<WAG>::new(&[0.05; 20], &[0.5; PIP_PARAM_N]).unwrap());
    correct_parameter_number(PIPModel::<HIVB>::new(&[0.05; 20], &[0.5; PIP_PARAM_N]).unwrap());
    correct_parameter_number(PIPModel::<BLOSUM>::new(&[0.05; 20], &[0.5; PIP_PARAM_N]).unwrap());
}
