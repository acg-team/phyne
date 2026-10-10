use std::path::Path;

use phylo::likelihood::ModelSearchCost;
use phylo::phylo_info::PhyloInfoBuilder;
use phylo::pip_model::{PIPCostBuilder, PIPModel};
use phylo::substitution_models::GTR;

#[test]

fn hiv_subset_valid_pip_likelihood() {
    let fldr = Path::new("./data/real_examples/");
    let alignment = fldr.join("HIV-1_env_DNA_mafft_alignment_subset.fasta");

    let info = PhyloInfoBuilder::new(alignment).build().unwrap();

    let pip = PIPModel::<GTR>::default();
    let c = PIPCostBuilder::new(pip, info).build().unwrap();
    let logl = c.cost();
    assert_ne!(logl, f64::NEG_INFINITY);
    assert!(logl < 0.0);
}
