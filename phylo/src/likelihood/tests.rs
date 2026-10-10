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
    dna_models::*, protein_models::*, QMatrix, SubstModel, SubstitutionCost,
    SubstitutionCostBuilder as SCB,
};
use crate::{tree, Error};

#[cfg(test)]
fn search_costs_equal_template<C: ModelSearchCost + TreeSearchCost>(cost: C) {
    assert_eq!(ModelSearchCost::cost(&cost), TreeSearchCost::cost(&cost));
}

#[cfg(test)]
fn setup_test_subst_cost<Q: QMatrix>(model: SubstModel<Q>) -> SubstitutionCost<Q, MSA> {
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

    SCB::new(model, info).build().unwrap()
}

#[test]
fn dna_search_costs_equal() {
    search_costs_equal_template(setup_test_subst_cost::<JC69>(SubstModel::<JC69>::default()));
    search_costs_equal_template(setup_test_subst_cost::<K80>(SubstModel::<K80>::default()));
    search_costs_equal_template(setup_test_subst_cost::<HKY>(
        SubstModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_subst_cost::<TN93>(
        SubstModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[0.5970915, 0.2940435]).unwrap(),
    ));
    search_costs_equal_template(setup_test_subst_cost::<GTR>(
        SubstModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    ));
}

#[test]
fn protein_search_costs_equal() {
    search_costs_equal_template(setup_test_subst_cost(SubstModel::<WAG>::default()));
    search_costs_equal_template(setup_test_subst_cost(SubstModel::<HIVB>::default()));
    search_costs_equal_template(setup_test_subst_cost(SubstModel::<BLOSUM>::default()));
    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(setup_test_subst_cost(
        SubstModel::<WAG>::new(freqs, &[]).unwrap(),
    ));
    search_costs_equal_template(setup_test_subst_cost(
        SubstModel::<HIVB>::new(freqs, &[]).unwrap(),
    ));
    search_costs_equal_template(setup_test_subst_cost(
        SubstModel::<BLOSUM>::new(freqs, &[]).unwrap(),
    ));
}

#[cfg(test)]
fn setup_test_pip_cost<Q: QMatrix>(model: PIPModel<Q>) -> PIPCost<Q, MSA> {
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

    PIPCB::new(model, info).build().unwrap()
}

#[test]
fn dna_pip_search_costs_equal() {
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<JC69>::with_default_substitution(&[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<K80>::with_default_substitution(&[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<HKY>::new(&[0.22, 0.26, 0.33, 0.19], &[1.2, 0.5, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<TN93>::new(&[0.22, 0.26, 0.33, 0.19], &[1.2, 0.5, 0.5970915, 0.2940435])
            .unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<GTR>::new(&[0.1, 0.3, 0.4, 0.2], &[1.2, 0.5, 5.0, 1.0, 1.0, 1.0, 1.0]).unwrap(),
    ));
}

#[test]
fn protein_pip_search_costs_equal() {
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<WAG>::with_default_substitution(&[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<HIVB>::with_default_substitution(&[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<BLOSUM>::with_default_substitution(&[1.2, 0.5]).unwrap(),
    ));

    let freqs = &[1.0 / 20.0; 20];
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<WAG>::new(freqs, &[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<HIVB>::new(freqs, &[1.2, 0.5]).unwrap(),
    ));
    search_costs_equal_template(setup_test_pip_cost(
        PIPModel::<BLOSUM>::new(freqs, &[1.2, 0.5]).unwrap(),
    ));
}

#[cfg(test)]
fn alphabet_mismatch_subst_model_template<Q: QMatrix>(
    model: SubstModel<Q>,
    alpha: &'static Alphabet,
) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let msa = MSA::from_aligned(Sequences::with_alphabet_unchecked(records, alpha), &tree).unwrap();
    let info = PhyloInfo { msa, tree };

    let res = SCB::new(model, info).build();

    assert_matches!(
        res,
        Err(Error::Alphabet(msg)) if msg.contains("alphabet mismatch")
    );
}

#[test]
fn alphabet_mismatch_subst_model_dna() {
    alphabet_mismatch_subst_model_template(SubstModel::<JC69>::default(), Alphabet::protein());
    alphabet_mismatch_subst_model_template(SubstModel::<K80>::default(), Alphabet::protein());
    alphabet_mismatch_subst_model_template(SubstModel::<HKY>::default(), Alphabet::protein());
    alphabet_mismatch_subst_model_template(SubstModel::<TN93>::default(), Alphabet::protein());
    alphabet_mismatch_subst_model_template(SubstModel::<GTR>::default(), Alphabet::protein());
}

#[test]
fn alphabet_mismatch_subst_model_protein() {
    alphabet_mismatch_subst_model_template(SubstModel::<WAG>::default(), Alphabet::dna());
    alphabet_mismatch_subst_model_template(SubstModel::<BLOSUM>::default(), Alphabet::dna());
    alphabet_mismatch_subst_model_template(SubstModel::<HIVB>::default(), Alphabet::dna());
}

#[cfg(test)]
fn alphabet_mismatch_subst_pip_template<Q: QMatrix>(model: PIPModel<Q>, alpha: &'static Alphabet) {
    // https://molevolworkshop.github.io/faculty/huelsenbeck/pdf/WoodsHoleHandout.pdf
    let fldr = Path::new("./data");
    let records = read_sequences(fldr.join("Huelsenbeck_example_long_DNA.fasta")).unwrap();
    let tree = tree!(&fs::read_to_string(fldr.join("Huelsenbeck_example.newick")).unwrap());
    let msa: MSA = MSA::from_aligned(
        Sequences::with_alphabet_unchecked(records.clone(), alpha),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { msa, tree };
    let res = PIPCB::new(model, info).build();

    assert_matches!(
        res,
        Err(Error::Alphabet(msg)) if msg.contains("alphabet mismatch")
    );
}

#[test]
fn alphabet_mismatch_pip_model() {
    alphabet_mismatch_subst_pip_template(PIPModel::<JC69>::default(), Alphabet::protein());
    alphabet_mismatch_subst_pip_template(PIPModel::<K80>::default(), Alphabet::protein());
    alphabet_mismatch_subst_pip_template(PIPModel::<HKY>::default(), Alphabet::protein());
    alphabet_mismatch_subst_pip_template(PIPModel::<TN93>::default(), Alphabet::protein());
    alphabet_mismatch_subst_pip_template(PIPModel::<GTR>::default(), Alphabet::protein());
    alphabet_mismatch_subst_pip_template(PIPModel::<WAG>::default(), Alphabet::dna());
    alphabet_mismatch_subst_pip_template(PIPModel::<BLOSUM>::default(), Alphabet::dna());
    alphabet_mismatch_subst_pip_template(PIPModel::<HIVB>::default(), Alphabet::dna());
}
