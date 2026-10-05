use std::num::NonZeroUsize;
use std::path::Path;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences, MSA};
use crate::alphabets::Alphabet;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{BranchOptimiser, StopCondition};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCostBuilder as PIPCB, PIPModel};
use crate::substitution_models::{dna_models::*, SubstModel, SubstitutionCostBuilder as SCB, WAG};
use crate::{record_wo_desc as record, tree, DEFAULT_EPSILON};

#[test]
fn branch_opt_likelihood_increase_pip() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPModel::<GTR>::with_default_substitution(&[14.142_1, 0.1414]).unwrap();
    let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), -5664.780425829528, epsilon = 1e-6);
    let o = BranchOptimiser::new(c.clone()).run().unwrap();

    assert!(o.final_cost > o.initial_cost);
    assert_eq!(c.cost(), o.initial_cost);

    let new_info = o.cost.info.clone();

    assert_ne!(new_info.tree.length, info.tree.length);
    assert_relative_eq!(
        new_info.tree.length,
        new_info.tree.iter().map(|n| n.blen).sum(),
        epsilon = 1e-4
    );

    let c = PIPCB::new(model, new_info).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);
    assert_eq!(o.costs[0], o.initial_cost);
    assert_eq!(o.costs.last().copied().unwrap(), o.final_cost);
    for i in 0..o.costs.len() - 1 {
        assert!(o.costs[i] <= o.costs[i + 1]);
    }
    assert!(o.costs[o.costs.len() - 1] - o.costs[o.costs.len() - 2] < DEFAULT_EPSILON);
}

#[test]
fn branch_opt_likelihood_increase_gtr() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let gtr = SubstModel::<GTR>::default();
    let o = BranchOptimiser::new(SCB::new(gtr.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();

    assert!(o.final_cost > o.initial_cost);
    assert_ne!(o.cost.tree().length, info.tree.length);

    let c = SCB::new(gtr, o.cost.info.clone()).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);

    assert_eq!(o.costs[0], o.initial_cost);
    assert_eq!(o.costs.last().copied().unwrap(), o.final_cost);
    for i in 0..o.costs.len() - 1 {
        assert!(o.costs[i] <= o.costs[i + 1]);
    }
    assert!(o.costs[o.costs.len() - 1] - o.costs[o.costs.len() - 2] < DEFAULT_EPSILON);
}

#[test]
fn branch_optimiser_against_phyml() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<JC69>::default();
    let o = BranchOptimiser::new(SCB::new(model.clone(), info.clone()).build().unwrap())
        .run()
        .unwrap();
    assert!(o.final_cost > o.initial_cost);

    let result_tree = o.cost.tree();
    assert_ne!(result_tree.length, info.tree.length);
    assert_relative_eq!(o.final_cost, -4086.56102, epsilon = 1e-4);
    let phyml_tree = tree!("((Gorilla:0.06683711,(Orangutan:0.21859880,Gibbon:0.31145586):0.06570906):0.03853171,Human:0.05356244,Chimpanzee:0.05417982);");
    for node in result_tree.leaves() {
        let phyml_node = phyml_tree.node(&phyml_tree.idx(&node.id));
        assert_relative_eq!(node.blen, phyml_node.blen, epsilon = 1e-4);
    }

    assert_eq!(result_tree.robinson_foulds(&info.tree), 0);
    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            result_tree.node(&result_tree.idx(taxon)).blen,
            phyml_tree.node(&phyml_tree.idx(taxon)).blen,
            epsilon = 1e-4
        );
    }
    assert_relative_eq!(result_tree.length, phyml_tree.length, epsilon = 1e-4);

    let c = SCB::new(model, o.cost.info.clone()).build().unwrap();
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_eq!(c.cost(), o.final_cost);
}

#[test]
fn repeated_optimisation_limit() {
    // This used to create -Inf likelihoods due to too long branch lengths and the probability
    // turning to 0.0.
    // This is supposed to run and not crash, no other conditions.

    let fldr = Path::new("./data/");
    let seq_file = fldr.join("p105.msa.fa");
    let info = PIB::new(seq_file).build().unwrap();

    let model = PIPModel::<WAG>::default();

    let mut cost = PIPCB::new(model, info).build().unwrap();
    let mut prev_cost = f64::NEG_INFINITY;
    let mut final_cost = TreeSearchCost::cost(&cost);
    let max_iterations = 100;
    let epsilon = 1e-5;

    let mut iterations = 0;
    while final_cost - prev_cost > epsilon && iterations < max_iterations {
        iterations += 1;
        prev_cost = final_cost;
        let branch_o = BranchOptimiser::new(cost.clone()).run().unwrap();
        assert!(branch_o.final_cost > branch_o.initial_cost);
        final_cost = branch_o.final_cost;
        cost = branch_o.cost;
    }
}

#[test]
fn only_gap_sequence() {
    let tree = tree!("((5207:0.8699783346462397,284812:226000000):0);");
    let msa: MSA = Alignment::from_aligned(
        Sequences::with_alphabet_unchecked(
            vec![record!("284812", b"-"), record!("5207", b"V")],
            Alphabet::protein(),
        ),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };
    let model = SubstModel::<WAG>::default();
    let c = SCB::new(model, info).build().unwrap();

    let o = BranchOptimiser::new(c).run().unwrap();
    assert!(o.final_cost >= o.initial_cost);
    assert!(o.final_cost.is_sign_negative());
}

#[test]
fn max_iter() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPModel::<GTR>::with_default_substitution(&[14.142_1, 0.1414]).unwrap();
    let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
    let unopt_cost = c.cost();
    let epsilon = DEFAULT_EPSILON;

    let res_default = BranchOptimiser::new(c.clone()).run().unwrap();
    assert_eq!(res_default.iterations, 4);
    let mut costs = res_default.costs;
    assert!(costs.pop().unwrap() - costs.pop().unwrap() < DEFAULT_EPSILON);

    let res = BranchOptimiser::with_stop_condition(
        c,
        StopCondition::max_iter_epsilon(NonZeroUsize::new(5).unwrap(), epsilon),
    )
    .run()
    .unwrap();

    // Basic expectations
    assert!(res.final_cost > res.initial_cost);
    assert!(res.final_cost >= unopt_cost);
    assert!(res.iterations <= 5);
    assert_eq!(res.initial_cost, unopt_cost);

    // Compare against default run, epsilon is the same, so results should be same
    assert_eq!(res.final_cost, res_default.final_cost);
    assert_eq!(res.iterations, res_default.iterations);
}

#[test]
fn precision() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPModel::<GTR>::with_default_substitution(&[14.142_1, 0.1414]).unwrap();
    let c = PIPCB::new(model.clone(), info.clone()).build().unwrap();
    let unopt_cost = c.cost();
    let epsilon = 1e-10;

    let res_default = BranchOptimiser::new(c.clone()).run().unwrap();

    let res = BranchOptimiser::with_stop_condition(c, StopCondition::epsilon(epsilon))
        .run()
        .unwrap();
    assert!(res.final_cost > res.initial_cost);

    assert!(res.final_cost >= unopt_cost);
    assert_eq!(res.initial_cost, unopt_cost);
    let mut costs = res.costs;
    assert!(costs.pop().unwrap() - costs.pop().unwrap() < epsilon);
    // Should take more iterations than default (which is 1e-3)
    assert!(res.iterations > res_default.iterations);
}

#[test]
fn fix_iter() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::default();
    let c = SCB::new(model, info).build().unwrap();
    let unopt_cost = c.cost();

    let res_default = BranchOptimiser::new(c.clone()).run().unwrap();
    assert_eq!(res_default.iterations, 4);

    let res = BranchOptimiser::with_stop_condition(
        c,
        StopCondition::fixed_iter(NonZeroUsize::new(8).unwrap()),
    )
    .run()
    .unwrap();
    assert!(res.final_cost > res.initial_cost);

    assert!(res.final_cost >= unopt_cost);
    assert!(res.final_cost >= res_default.final_cost);
    assert_eq!(res.iterations, 8);
    assert_eq!(res.initial_cost, unopt_cost);
}
