use std::path::Path;

use phylo::likelihood::ModelSearchCost;
use phylo::phylo_info::PhyloInfoBuilder;
use phylo::substitution_models::{SubstModel, SubstitutionCostBuilder, GTR};

#[test]
fn hiv_subset_valid_subst_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");
    let info = PhyloInfoBuilder::new(alignment).build().unwrap();
    let gtr = SubstModel::<GTR>::default();
    let c = SubstitutionCostBuilder::new(gtr, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}
