use std::fs;
use std::path::Path;

use assert_matches::assert_matches;

use crate::alignment::{Alignment, Sequences, MSA};
use crate::alphabets::Alphabet;
use crate::io::read_sequences;
use crate::likelihood::{ModelSearchCost, TreeSearchCost};
use crate::phylo_info::PhyloInfo;
use crate::pip_model::{PIPCost, PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, QMatrix, QMatrixMaker, SubstModel, SubstitutionCost,
    SubstitutionCostBuilder as SCB,
};
use crate::{tree, Error};

#[cfg(test)]
fn search_costs_equal_template<C: ModelSearchCost + TreeSearchCost>(cost: C) {
    assert_eq!(ModelSearchCost::cost(&cost), TreeSearchCost::cost(&cost));
}

#[cfg(test)]
fn test_subst_model<Q: QMatrix + QMatrixMaker>(
    freqs: &[f64],
    params: &[f64],
) -> SubstitutionCost<Q, MSA> {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf

    let fldr = Path::new("./data");
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let msa = Alignment::from_aligned(
        Sequences::with_alphabet_unchecked(records.clone(), Q::alphabet()),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    SCB::new(model, info).build().unwrap()
}

#[test]
fn dna_search_costs_equal() {
    search_costs_equal_template(test_subst_model::<JC69>(&[], &[]));
    search_costs_equal_template(test_subst_model::<K80>(&[], &[2.0]));
    search_costs_equal_template(test_subst_model::<HKY>(&[0.22, 0.26, 0.33, 0.19], &[0.5]));
    search_costs_equal_template(test_subst_model::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    ));
    search_costs_equal_template(test_subst_model::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    ));
}

#[test]
fn protein_search_costs_equal() {
    search_costs_equal_template(test_subst_model::<WAG>(&[], &[]));
    search_costs_equal_template(test_subst_model::<HIVB>(&[], &[]));
    search_costs_equal_template(test_subst_model::<BLOSUM>(&[], &[]));
    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(test_subst_model::<WAG>(freqs, &[]));
    search_costs_equal_template(test_subst_model::<HIVB>(freqs, &[]));
    search_costs_equal_template(test_subst_model::<BLOSUM>(freqs, &[]));
}

#[cfg(test)]
fn test_pip_model<Q: QMatrix + QMatrixMaker>(freqs: &[f64], params: &[f64]) -> PIPCost<Q, MSA> {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();

    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let msa = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(records.clone(), Q::alphabet()),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let model = PIPModel::<Q>::new(freqs, params).unwrap();
    PIPCB::new(model, info).build().unwrap()
}

#[test]
fn dna_pip_search_costs_equal() {
    search_costs_equal_template(test_pip_model::<JC69>(&[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<K80>(&[], &[1.2, 0.5, 2.0]));
    search_costs_equal_template(test_pip_model::<HKY>(
        &[0.22, 0.26, 0.33, 0.19],
        &[1.2, 0.5, 0.5],
    ));
    search_costs_equal_template(test_pip_model::<TN93>(
        &[0.22, 0.26, 0.33, 0.19],
        &[1.2, 0.5, 0.5970915, 0.2940435, 0.00135],
    ));
    search_costs_equal_template(test_pip_model::<GTR>(
        &[0.1, 0.3, 0.4, 0.2],
        &[1.2, 0.5, 5.0, 1.0, 1.0, 1.0, 1.0, 5.0],
    ));
}

#[test]
fn protein_pip_search_costs_equal() {
    search_costs_equal_template(test_pip_model::<WAG>(&[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<HIVB>(&[], &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<BLOSUM>(&[], &[1.2, 0.5]));
    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(test_pip_model::<WAG>(freqs, &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<HIVB>(freqs, &[1.2, 0.5]));
    search_costs_equal_template(test_pip_model::<BLOSUM>(freqs, &[1.2, 0.5]));
}

#[cfg(test)]
fn alphabet_mismatch_subst_model_template<Q: QMatrix + QMatrixMaker>(
    alpha: &'static Alphabet,
    freqs: &[f64],
    params: &[f64],
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let msa = MSA::from_aligned(Sequences::with_alphabet_unchecked(records, alpha), &tree).unwrap();
    let info = PhyloInfo { msa, tree };

    let model = SubstModel::<Q>::new(freqs, params).unwrap();
    let res = SCB::new(model, info).build();

    assert_matches!(
        res,
        Err(Error::Alphabet(msg)) if msg.contains("alphabet mismatch")
    );
}

#[test]
fn alphabet_mismatch_subst_model() {
    alphabet_mismatch_subst_model_template::<JC69>(Alphabet::protein(), &[], &[]);
    alphabet_mismatch_subst_model_template::<K80>(Alphabet::protein(), &[], &[2.0]);
    alphabet_mismatch_subst_model_template::<HKY>(
        Alphabet::protein(),
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5],
    );
    alphabet_mismatch_subst_model_template::<TN93>(
        Alphabet::protein(),
        &[0.22, 0.26, 0.33, 0.19],
        &[0.5970915, 0.2940435, 0.00135],
    );
    alphabet_mismatch_subst_model_template::<GTR>(
        Alphabet::protein(),
        &[0.1, 0.3, 0.4, 0.2],
        &[1.5; 5],
    );
    alphabet_mismatch_subst_model_template::<WAG>(Alphabet::dna(), &[], &[]);
    alphabet_mismatch_subst_model_template::<BLOSUM>(Alphabet::dna(), &[], &[]);
    alphabet_mismatch_subst_model_template::<HIVB>(Alphabet::dna(), &[], &[]);
}

#[cfg(test)]
fn alphabet_mismatch_subst_pip_template<Q: QMatrix + QMatrixMaker>(
    alpha: &'static Alphabet,
    freqs: &[f64],
    params: &[f64],
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let msa = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(records.clone(), alpha),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };
    let model = PIPModel::<Q>::new(freqs, params).unwrap();
    let res = PIPCB::new(model, info).build();

    assert_matches!(
        res,
        Err(Error::Alphabet(msg)) if msg.contains("alphabet mismatch")
    );
}

#[test]
fn alphabet_mismatch_pip_model() {
    alphabet_mismatch_subst_pip_template::<JC69>(Alphabet::protein(), &[], &[1.3, 0.5]);
    alphabet_mismatch_subst_pip_template::<K80>(Alphabet::protein(), &[], &[1.3, 0.5, 2.0]);
    alphabet_mismatch_subst_pip_template::<HKY>(
        Alphabet::protein(),
        &[0.22, 0.26, 0.33, 0.19],
        &[1.3, 0.5, 0.5],
    );
    alphabet_mismatch_subst_pip_template::<TN93>(
        Alphabet::protein(),
        &[0.22, 0.26, 0.33, 0.19],
        &[1.3, 0.5, 0.5970915, 0.2940435, 0.00135],
    );
    alphabet_mismatch_subst_pip_template::<GTR>(
        Alphabet::protein(),
        &[0.1, 0.3, 0.4, 0.2],
        &[1.5; 7],
    );
    alphabet_mismatch_subst_pip_template::<WAG>(Alphabet::dna(), &[], &[1.3, 0.5]);
    alphabet_mismatch_subst_pip_template::<BLOSUM>(Alphabet::dna(), &[], &[1.3, 0.5]);
    alphabet_mismatch_subst_pip_template::<HIVB>(Alphabet::dna(), &[], &[1.3, 0.5]);
}
