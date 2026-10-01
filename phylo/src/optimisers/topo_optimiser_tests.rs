use std::num::NonZeroUsize;
use std::path::Path;

use approx::assert_relative_eq;

use crate::alignment::{Alignment, Sequences, MSA};
use crate::evolutionary_models::FrequencyOptimisation::Empirical;
use crate::likelihood::TreeSearchCost;
use crate::optimisers::{
    BranchOptimiser, ModelOptimiser, NniOptimiser, SprOptimiser, StopCondition, TopologyOptimiser,
    TreeOptimisationResult,
};
use crate::parsimony::{scoring::ModelScoringBuilder, BasicParsimonyCost, DolloParsimonyCost};
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::pip_model::{PIPCost, PIPCostBuilder as PIPCB, PIPModel};
use crate::random::{FakeGenerator, FakeRng};
use crate::substitution_models::{
    dna_models::*, protein_models::*, QMatrix, SubstModel, SubstitutionCost,
    SubstitutionCostBuilder as SCB,
};
use crate::tkf_model::{blocks_of_alignment, TKF92CostBuilder, TKF92IndelAddBlocksCostBuilder};
use crate::{record_wo_desc as record, tree};

// Macros for tests where precomputed results can be used to speed up local testing
macro_rules! define_optimise_trees {
    ($($fn_name:ident: { model = $model:ident, cost = $cost:ident, builder = $builder:ident , move_optimiser = $move_optimiser:ident }),* $(,)?) => {
        $(
            fn $fn_name<Q: QMatrix + Send>(
                seq_file: &std::path::Path,
                _tree_precomputed_file: &std::path::Path,
                model: $model<Q>,
                rng: &mut FakeGenerator,
            ) -> TreeOptimisationResult<$cost<Q, MSA>> {
                let start_info = PIB::new(seq_file).build_w_rng(rng).unwrap();
                let initial_cost = $builder::new(model.clone(), start_info).build().unwrap();
cfg_if::cfg_if! {
if #[cfg(not(feature="precomputed-test-results"))] {
                TopologyOptimiser::new(initial_cost, $move_optimiser {}, rng).run().unwrap()
} else {
                if let Ok(precomputed) =
                    PIB::with_attrs(seq_file, _tree_precomputed_file).build()
                {
                    let final_cost = $builder::new(model, precomputed.clone()).build().unwrap();

                    TreeOptimisationResult {
                        initial_cost: initial_cost.cost(),
                        final_cost: final_cost.cost(),
                        iterations: 0,
                        cost: final_cost.clone(),
                        costs: vec![initial_cost.cost(), final_cost.cost()],
                    }
                } else {
                    let res = TopologyOptimiser::new(initial_cost, $move_optimiser {}, rng).run().unwrap();
                    assert!(crate::io::write_newick_to_file(
                        &[res.cost.tree().clone()],
                        _tree_precomputed_file
                    )
                    .is_ok());
                    res
                }
}}
            }
        )*
    };
}

define_optimise_trees!(
    optimise_tree: { model = SubstModel, cost = SubstitutionCost, builder = SCB, move_optimiser = SprOptimiser },
    optimise_tree_pip: { model = PIPModel, cost = PIPCost, builder = PIPCB, move_optimiser = SprOptimiser },
    optimise_tree_nni: { model = SubstModel, cost = SubstitutionCost, builder = SCB, move_optimiser = NniOptimiser },
);

#[test]
fn k80_simple() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"CTATATATAC"),
            record!("B", b"ATATATATAA"),
            record!("C", b"TTATATATAT"),
            record!("D", b"TTATATATAT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_simple_nni() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let tree = tree!("(((A:1.0,B:1.0)E:2.0,(C:1.0,D:1.0)F:2.0)G:3.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"CTATATATAC"),
            record!("B", b"ATATATATAA"),
            record!("C", b"TTATATATAT"),
            record!("D", b"TTATATATAT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, NniOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_given() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from a given tree
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::with_attrs(fldr.join("K80.fasta"), fldr.join("../tree.newick"))
        .build()
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
fn k80_sim_data_from_nj() {
    // Check that optimisation on k80 data improves k80 likelihood when starting from an NJ tree
    let mut fake_rng = FakeGenerator::default();
    let fldr = Path::new("./data/sim/K80");
    let info = PIB::new(fldr.join("K80.fasta"))
        .build_w_rng(&mut fake_rng)
        .unwrap();
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let c = SCB::new(k80.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut fake_rng)
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, -4060.91964, epsilon = 1e-5);

    let c = SCB::new(k80, o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn k80_sim_data_vs_phyml() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build_w_rng(&mut FakeGenerator::default())
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let c = SCB::new(jc69.clone(), o.cost.info.clone()).build().unwrap();
    assert_relative_eq!(c.cost(), o.final_cost);
    assert_relative_eq!(c.cost(), o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();

    let phyml_c = SCB::new(jc69, phyml_info.clone()).build().unwrap();
    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);
    assert_relative_eq!(tree.length, phyml_info.tree.length, epsilon = 1e-4);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Human", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.node(&tree.idx(taxon)).blen,
            phyml_info.tree.node(&phyml_info.tree.idx(taxon)).blen,
            epsilon = 1e-5
        );
    }
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn k80_sim_data_vs_phyml_wrong_start() {
    // Check that optimisation on k80 data under JC69 produces similar tree to PhyML with matching likelihood
    // when starting from a wrong tree
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let jc69 = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let c = SCB::new(jc69.clone(), info).build().unwrap();
    let unopt_logl = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_logl);
    assert_eq!(o.initial_cost, unopt_logl);
    assert_eq!(o.final_cost, o.cost.cost());

    let phyml_info = PIB::with_attrs(
        fldr.join("K80/K80.fasta"),
        fldr.join("K80/phyml_tree.newick"),
    )
    .build()
    .unwrap();
    let phyml_c = SCB::new(jc69.clone(), phyml_info.clone()).build().unwrap();

    assert_relative_eq!(o.final_cost, phyml_c.cost(), epsilon = 1e-5);
    assert_relative_eq!(o.final_cost, -4038.721121221992, epsilon = 1e-5);

    let tree = o.cost.tree();
    assert_relative_eq!(tree.length, phyml_info.tree.length, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_info.tree), 0);

    let taxa = ["Gorilla", "Orangutan", "Gibbon", "Chimpanzee"];
    for taxon in taxa.iter() {
        assert_relative_eq!(
            tree.by_id(taxon).blen,
            phyml_info.tree.by_id(taxon).blen,
            epsilon = 1e-5
        );
    }
    let human_idx = tree.idx("Human");
    assert_relative_eq!(
        tree.by_id("Human").blen + tree.node(&tree.sibling(&human_idx).unwrap()).blen,
        phyml_info.tree.by_id("Human").blen,
        epsilon = 1e-4
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_no_gaps_vs_phyml_nj_tree_start_nni() {
    // This test only passes because the NNI moves dont run into a local optimum and therefore
    // find the same best tree as the SprOptimiser. So, if the data changes the test may fail.

    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps starting from an NJ tree
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");
    let rng = &mut FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let res = optimise_tree_nni(&seq_file, &tree_file, wag.clone(), rng);
    assert!(res.final_cost >= res.initial_cost);
    let wag_tree = res.cost.tree();

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(wag, phyml_res.clone()).build().unwrap().cost();

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(wag_tree.length, phyml_res.tree.length, epsilon = 1e-4);
    assert_eq!(wag_tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(res.final_cost, phyml_logl, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_no_gaps_vs_phyml_nj_tree_start_spr() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // on sequences without gaps starting from an NJ tree
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");
    let rng = &mut FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let res = optimise_tree(&seq_file, &tree_file, wag.clone(), rng);
    assert!(res.final_cost >= res.initial_cost);
    let wag_tree = res.cost.tree();

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_nogap.newick"))
        .build()
        .unwrap();
    let phyml_logl = SCB::new(wag, phyml_res.clone()).build().unwrap().cost();

    // Compare tree height and logl to the output of PhyML
    assert_relative_eq!(wag_tree.length, phyml_res.tree.length, epsilon = 1e-4);
    assert_eq!(wag_tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(res.final_cost, phyml_logl, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn test_nni_and_spr_find_same_tree() {
    // This test only passes because the NNI moves dont run into a local optimum and therefore
    // find the same best tree as the SprOptimiser. So, if the data changes the test may fail.
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let tree_file = fldr.join("jati_wag_nogap_nj_start.newick");
    let rng = &mut FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let res_nni = optimise_tree_nni(&seq_file, &tree_file, wag.clone(), rng);
    let res_spr = optimise_tree(&seq_file, &tree_file, wag.clone(), rng);
    assert!(res_nni.final_cost >= res_nni.initial_cost);
    assert!(res_spr.final_cost >= res_spr.initial_cost);
    assert_relative_eq!(res_nni.final_cost, res_spr.final_cost, epsilon = 1e-5);

    let nni_tree = res_nni.cost.tree();
    let spr_tree = res_spr.cost.tree();

    assert_relative_eq!(nni_tree.length, spr_tree.length, epsilon = 1e-4);
    assert_eq!(nni_tree.robinson_foulds(spr_tree), 0);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_vs_subst_dna_tree() {
    // Check that optimisation on k80 data under PIP and substitution model produces similar trees
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("wrong_tree.newick"))
        .build()
        .unwrap();
    let rng = &mut FakeGenerator::default();

    let k80 = SubstModel::<K80>::new(&[], &[4.0]).unwrap();
    let k80_res = TopologyOptimiser::new(
        SCB::new(k80.clone(), info.clone()).build().unwrap(),
        SprOptimiser {},
        rng,
    )
    .run()
    .unwrap();

    let pip = PIPModel::<K80>::new(&[], &[0.5, 0.4, 4.0]).unwrap();
    let pip_res = TopologyOptimiser::new(
        PIPCB::new(pip.clone(), info).build().unwrap(),
        SprOptimiser {},
        rng,
    )
    .run()
    .unwrap();

    // Tree topologies under PIP+K80 and K80 should match
    assert_eq!(pip_res.cost.tree().robinson_foulds(k80_res.cost.tree()), 0);

    // Check that likelihoods under substitution model are similar for both trees
    // but reoptimise branch lengths for PIP tree because they are not comparable
    let k80_pip_tree_res = BranchOptimiser::new(SCB::new(k80, pip_res.cost.info).build().unwrap())
        .run()
        .unwrap();
    assert_relative_eq!(
        k80_res.final_cost,
        k80_pip_tree_res.final_cost,
        epsilon = 1e-6
    );

    // Check that likelihoods under PIP are similar for both trees
    // but reoptimise branch lengths for substitution tree because they are not comparable
    let pip_k80_tree_res =
        BranchOptimiser::new(PIPCB::new(pip, k80_res.cost.info).build().unwrap())
            .run()
            .unwrap();
    assert_relative_eq!(
        pip_res.final_cost,
        pip_k80_tree_res.final_cost,
        epsilon = 1e-6
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_nogaps_pip_vs_subst_tree_nj_start() {
    // Check that optimisation on protein data under WAG produces similar trees for PIP and substitution model
    // on sequences without gaps
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");

    let pip = PIPModel::<WAG>::new(&[], &[50.0, 0.1]).unwrap();
    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let rng = &mut FakeGenerator::default();

    let pip_res = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nogap_pip_vs_wag.newick"),
        pip.clone(),
        rng,
    );
    assert!(pip_res.final_cost >= pip_res.initial_cost);
    let pip_tree = pip_res.cost.tree();

    let wag_res = optimise_tree(
        &seq_file,
        &fldr.join("jati_wag_nogap_pip_vs_wag.newick"),
        wag.clone(),
        rng,
    );
    assert!(wag_res.final_cost >= wag_res.initial_cost);
    let wag_tree = wag_res.cost.tree();

    // Compare tree created with a substitution model to the one with PIP
    assert_eq!(pip_tree.robinson_foulds(wag_tree), 0);

    // Check that likelihoods under same model are similar for both trees
    let pip_tree_reopt_wag_logl = BranchOptimiser::new(
        SCB::new(wag.clone(), pip_res.cost.info.clone())
            .build()
            .unwrap(),
    )
    .run()
    .unwrap()
    .final_cost;
    assert_relative_eq!(wag_res.final_cost, pip_tree_reopt_wag_logl, epsilon = 1e-5);

    // Check that the likelihoods under the same model are similar for both trees
    let wag_tree_reopt_pip_logl =
        BranchOptimiser::new(PIPCB::new(pip.clone(), wag_res.cost.info).build().unwrap())
            .run()
            .unwrap()
            .final_cost;
    assert_relative_eq!(pip_res.final_cost, wag_tree_reopt_pip_logl, epsilon = 1e-5);

    // Just a random check that reestimating branch lengths makes no difference
    let new_pip_cost = PIPCB::new(pip, pip_res.cost.info.clone()).build().unwrap();
    let reopt_res = BranchOptimiser::new(new_pip_cost.clone()).run().unwrap();
    assert_relative_eq!(new_pip_cost.cost(), reopt_res.final_cost, epsilon = 1e-5);
    assert_relative_eq!(
        pip_tree.length,
        reopt_res.cost.info.tree.length,
        epsilon = 1e-5
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_optimise_model_tree() {
    // Check that tree optimisation under PIP has a better likelihood when the model is also optimised
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");

    let rng = &mut FakeGenerator::from_rng(FakeRng::from_u64_values(vec![u64::MAX]));
    let start_info = PIB::new(seq_file.clone()).build_w_rng(rng).unwrap();

    // Optimise tree starting from an NJ tree and initial model
    let pip = PIPModel::<WAG>::new(&[], &[1.4, 0.5]).unwrap();
    let tree_opt_result = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nj_start.newick"),
        pip.clone(),
        rng,
    );
    assert!(tree_opt_result.final_cost >= tree_opt_result.initial_cost);

    // Optimise model parameters on the starting tree
    let model_opt_result = ModelOptimiser::new(
        PIPCB::new(pip.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();
    let optimised_pip = model_opt_result.cost.model;

    assert!(model_opt_result.final_cost >= model_opt_result.initial_cost);
    assert!(model_opt_result.final_cost >= tree_opt_result.final_cost);

    // Optimise tree again with the optimised model
    let model_tree_opt_result = optimise_tree_pip(
        &seq_file,
        &fldr.join("jati_pip_nj_start_model_opt.newick"),
        optimised_pip.clone(),
        rng,
    );

    assert!(model_tree_opt_result.final_cost >= model_opt_result.final_cost);
    assert!(model_tree_opt_result.final_cost >= tree_opt_result.initial_cost);

    assert_eq!(
        model_tree_opt_result
            .cost
            .tree()
            .robinson_foulds(tree_opt_result.cost.tree()),
        0
    );
    assert!(model_tree_opt_result.final_cost > tree_opt_result.final_cost);

    // Check that the optimised tree+model likelihood is higher than the likelihood of thefirst optimised tree (no model optimisation) with the optimised model
    let tree_opt_optimised_model_logl =
        PIPCB::new(optimised_pip, tree_opt_result.cost.info.clone())
            .build()
            .unwrap()
            .cost();
    assert!(model_tree_opt_result.final_cost > tree_opt_optimised_model_logl);

    // Check that the optimised tree+model likelihood is higher than the likelihood of the same tree with the starting model
    let model_tree_opt_start_model_logl = PIPCB::new(pip, model_tree_opt_result.cost.info.clone())
        .build()
        .unwrap()
        .cost();
    assert!(model_tree_opt_result.final_cost > model_tree_opt_start_model_logl);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_empirical_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using empirical frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let starting_tree = fldr.join("phyml_nj_tree.newick");
    let start_info = PIB::with_attrs(seq_file.clone(), starting_tree)
        .build()
        .unwrap();

    let rng = &mut FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let o = ModelOptimiser::new(
        SCB::new(wag.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();

    assert!(o.final_cost >= o.initial_cost);
    let wag_opt = o.cost.model;

    let res = TopologyOptimiser::new(
        SCB::new(wag_opt.clone(), start_info).build().unwrap(),
        SprOptimiser {},
        rng,
    )
    .run()
    .unwrap();

    let tree = res.cost.tree();
    assert!(res.final_cost >= res.initial_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();

    assert_relative_eq!(tree.length, phyml_res.tree.length, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_res.tree), 0);

    let wag_opt_phyml_logl = SCB::new(wag_opt, phyml_res).build().unwrap().cost();
    assert_relative_eq!(res.final_cost, wag_opt_phyml_logl, epsilon = 1e-5);
    assert_relative_eq!(res.final_cost, -5258.79254297163, epsilon = 1e-5);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_wag_vs_phyml_empirical_freqs() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let tree_file = fldr.join("jati_pip_wag_empirical.newick");

    let start_info = PIB::new(seq_file.clone())
        .build_w_rng(&mut FakeGenerator::default())
        .unwrap();
    let pip = PIPModel::<WAG>::new(&[], &[1.0, 2.0]).unwrap();
    // Use empirical frequencies and optimise lambda and mu for PIP
    let o = ModelOptimiser::new(
        PIPCB::new(pip.clone(), start_info.clone()).build().unwrap(),
        Empirical,
    )
    .run()
    .unwrap();

    let rng = &mut FakeGenerator::default();
    let pip_opt = o.cost.model;
    let res = optimise_tree_pip(&seq_file, &tree_file, pip_opt.clone(), rng);

    // Optimised tree should be better than the starting tree
    assert!(res.final_cost >= o.final_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_empirical.newick"))
        .build()
        .unwrap();
    // Reoptimise branch lengths for the phyml tree because they are not comparable with PIP
    let phyml_brlen_opt = BranchOptimiser::new(PIPCB::new(pip_opt, phyml_res).build().unwrap())
        .run()
        .unwrap();

    // Check that our tree is better than phyml
    assert!(res.final_cost > phyml_brlen_opt.final_cost);
    assert_relative_eq!(
        res.cost.tree().length,
        phyml_brlen_opt.cost.info.tree.length,
        epsilon = 1e-2
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn wag_vs_phyml_fixed_freqs() {
    // Check that optimisation on protein data under WAG produces similar tree to PhyML with matching likelihoods
    // when using fixed frequencies
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let starting_tree = fldr.join("phyml_nj_tree.newick");
    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let start_info = PIB::with_attrs(seq_file.clone(), starting_tree)
        .build()
        .unwrap();

    let rng = &mut FakeGenerator::default();

    let res = TopologyOptimiser::new(
        SCB::new(wag.clone(), start_info).build().unwrap(),
        SprOptimiser {},
        rng,
    )
    .run()
    .unwrap();

    assert_relative_eq!(res.initial_cost, -5311.776363282737, epsilon = 1e-3);
    assert!(res.final_cost >= res.initial_cost);

    let phyml_res = PIB::with_attrs(seq_file.clone(), fldr.join("phyml_wag_fixed.newick"))
        .build()
        .unwrap();

    let tree = res.cost.tree();
    assert_relative_eq!(tree.length, phyml_res.tree.length, epsilon = 1e-4);
    assert_eq!(tree.robinson_foulds(&phyml_res.tree), 0);
    assert_relative_eq!(
        res.final_cost,
        SCB::new(wag, phyml_res).build().unwrap().cost(),
        epsilon = 1e-5
    );

    assert_relative_eq!(res.final_cost, -5295.08423, epsilon = 1e-3);
}

#[test]
fn basic_parsimony_tree_search() {
    let seqs = Sequences::new_unchecked(vec![
        record!("A", b"GGA"),
        record!("B", b"GGG"),
        record!("C", b"ACA"),
        record!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -6.0);

    let res = TopologyOptimiser::new(cost, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();
    assert_eq!(res.final_cost, -4.0);
}

#[test]
fn dollo_tree_search() {
    let tree = tree!("(((A:1.0,C:1.0)E:2.0,(D:1.0,B:1.0)F:2.0)G:3.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"TTTTTTTTTTTCTATATATA-"),
            record!("B", b"TTTTTTTTTTTATATATAT-A"),
            record!("C", b"GGGGGGGGGGGTTATATATA-"),
            record!("D", b"GGGGGGGGGGGTTATATATA-"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { msa, tree };
    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let scoring = ModelScoringBuilder::new(k80)
        .times(vec![1.0])
        .build()
        .unwrap();

    let c = DolloParsimonyCost::with_scoring(info, scoring);
    let unopt_score = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn dollo_tree_search_sim_data_simple() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let c = DolloParsimonyCost::new(info);
    let unopt_score = c.cost();

    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();
    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn dollo_tree_search_sim_data_model() {
    let fldr = Path::new("./data/sim/");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();

    let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
    let scoring = ModelScoringBuilder::new(k80)
        .times(vec![1.0])
        .build()
        .unwrap();

    let c = DolloParsimonyCost::with_scoring(info, scoring);
    let unopt_score = c.cost();
    let o = TopologyOptimiser::new(c, SprOptimiser {}, &mut FakeGenerator::default())
        .run()
        .unwrap();

    assert!(o.final_cost >= unopt_score);
    assert_eq!(o.initial_cost, unopt_score);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn fix_iter_low() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let mut rng = FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::new(seq_file).build_w_rng(&mut rng).unwrap();
    let c = SCB::new(wag, info).build().unwrap();
    let unopt_cost = c.cost();

    let res_default = TopologyOptimiser::new(c.clone(), SprOptimiser {}, &mut rng)
        .run()
        .unwrap();
    // Without the stop condition and with the fake rng will run deterministically for 2 iterations
    assert_eq!(res_default.iterations, 2);

    // With the stop condition will run for exactly 1 iteration
    let result = TopologyOptimiser::with_stop_condition(
        c,
        SprOptimiser {},
        &mut rng,
        StopCondition::fixed_iter(NonZeroUsize::new(1).unwrap()),
    )
    .run()
    .unwrap();
    assert!(result.final_cost >= unopt_cost);
    assert_eq!(result.iterations, 1);
    assert_eq!(result.initial_cost, unopt_cost);

    assert!(result.final_cost <= res_default.final_cost);
}

#[test]
fn precision() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("seqs.fasta");
    let mut rng = FakeGenerator::default();
    let epsilon = 1e-1;

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::new(seq_file).build_w_rng(&mut rng).unwrap();
    let c = SCB::new(wag, info).build().unwrap();
    let unopt_cost = c.cost();

    let res_default = TopologyOptimiser::new(c.clone(), SprOptimiser {}, &mut rng)
        .run()
        .unwrap();
    // Without the stop condition and with the fake rng will run deterministically for 2 iterations
    assert_eq!(res_default.iterations, 2);

    // With the stop condition will run for less iterations because epsilon is high
    let result = TopologyOptimiser::with_stop_condition(
        c,
        SprOptimiser {},
        &mut rng,
        StopCondition::epsilon(epsilon),
    )
    .run()
    .unwrap();
    assert!(result.final_cost >= unopt_cost);

    let mut costs = result.costs;
    assert!(costs.pop().unwrap() - costs.pop().unwrap() < epsilon);
    assert!(result.iterations <= res_default.iterations);
    assert_eq!(result.initial_cost, unopt_cost);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn fix_iter() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let mut rng = FakeGenerator::default();

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::new(seq_file).build_w_rng(&mut rng).unwrap();
    let c = SCB::new(wag, info).build().unwrap();
    let unopt_cost = c.cost();

    let res_default = TopologyOptimiser::new(c.clone(), SprOptimiser {}, &mut rng)
        .run()
        .unwrap();
    // Without the stop condition and with the fake rng will run deterministically for 2 iterations
    assert_eq!(res_default.iterations, 2);

    // With the stop condition will run for exactly 6 iterations
    let result = TopologyOptimiser::with_stop_condition(
        c,
        SprOptimiser {},
        &mut rng,
        StopCondition::fixed_iter(NonZeroUsize::new(6).unwrap()),
    )
    .run()
    .unwrap();
    assert!(result.final_cost >= unopt_cost);
    assert!(result.final_cost >= res_default.final_cost);
    assert_eq!(result.iterations, 6);
    assert_eq!(result.initial_cost, unopt_cost);
}

#[test]
fn max_iter() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let seq_file = fldr.join("nogap_seqs.fasta");
    let epsilon = 1e-10;

    let wag = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let info = PIB::new(seq_file)
        .build_w_rng(&mut FakeGenerator::default())
        .unwrap();
    let c = SCB::new(wag, info).build().unwrap();
    let unopt_cost = c.cost();

    let res_default =
        TopologyOptimiser::new(c.clone(), SprOptimiser {}, &mut FakeGenerator::default())
            .run()
            .unwrap();
    // Without the stop condition and with the fake rng will run deterministically for 2 iterations
    assert_eq!(res_default.iterations, 2);

    // With the stop condition will run for at most 5 iterations, and would continue running because the epsilon is very low
    let result = TopologyOptimiser::with_stop_condition(
        c,
        SprOptimiser {},
        &mut FakeGenerator::default(),
        StopCondition::max_iter_epsilon(NonZeroUsize::new(3).unwrap(), epsilon),
    )
    .run()
    .unwrap();
    assert!(result.final_cost >= unopt_cost);
    assert!(result.iterations <= 3);
    assert!(result.final_cost >= res_default.final_cost);
    assert_eq!(result.initial_cost, unopt_cost);
}

// Warning: If you change this test, please also update the example in the README.md file
// accordingly!
#[test]
fn example_main_from_readme() {
    use crate::likelihood::TreeSearchCost;
    use crate::optimisers::TopologyOptimiser;
    use crate::phylo_info::PhyloInfoBuilder;
    use crate::substitution_models::{SubstModel, SubstitutionCostBuilder, K80};
    use crate::Result;

    fn main() -> Result<()> {
        // Note: This example uses test data from the repository
        let info = PhyloInfoBuilder::new("./examples/data/K80.fasta").build()?;
        let k80 = SubstModel::<K80>::new(&[], &[4.0, 1.0]).unwrap();
        let c = SubstitutionCostBuilder::new(k80, info).build()?;
        let unopt_cost = c.cost();
        let mut rng = FakeGenerator::default();
        let optimiser = TopologyOptimiser::new(c, SprOptimiser {}, &mut rng);
        let result = optimiser.run()?;
        assert_eq!(unopt_cost, result.initial_cost);
        assert!(result.final_cost > result.initial_cost);
        assert!(result.iterations <= 100);
        assert_eq!(result.cost.tree().len(), 9); // The initial tree has 9 nodes, 5 leaves and 4 internal nodes, and so should the resulting tree.
        Ok(())
    }
    main().unwrap();
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn tkf92_topo_opti() {
    let dir = Path::new("data/tkf/reestimate/");
    let msa = dir.join("masa.fasta");
    let tree = dir.join("tree.newick");
    let phylo = PIB::with_attrs(msa, tree).build_with_ancestors().unwrap();
    let subst_model = SubstModel::<GTR>::new(&[], &[]).unwrap();
    let lambda = 0.1;
    let mu = 0.2;
    let r = 0.3;
    let tkf_cost = TKF92CostBuilder::new(&[lambda, mu, r], subst_model.clone(), phylo.clone())
        .build()
        .unwrap();
    let unopt_cost = tkf_cost.cost();
    let mut prev_cost = tkf_cost.clone();
    assert_ne!(unopt_cost, f64::NEG_INFINITY);
    let mut rng = FakeGenerator::default();
    let initial_msa_blocking = blocks_of_alignment(&phylo.msa);
    for _ in 0..5 {
        let result = TopologyOptimiser::with_stop_condition(
            prev_cost.clone(),
            NniOptimiser {},
            &mut rng,
            StopCondition::max_iter_epsilon(NonZeroUsize::new(1).unwrap(), 1e-2),
        )
        .run()
        .unwrap();
        let masa = result.cost.masa().clone();
        let new_phylo = PhyloInfo {
            msa: masa,
            tree: result.cost.tree().clone(),
        };
        let new_indel_cost = TKF92IndelAddBlocksCostBuilder::new(
            &[lambda, mu, r],
            initial_msa_blocking.clone(),
            new_phylo.clone(),
        )
        .build()
        .unwrap();
        let new_subst_cost = SCB::new(subst_model.clone(), new_phylo).build().unwrap();
        let new_cost = new_indel_cost.cost() + new_subst_cost.cost();

        prev_cost = result.cost.clone();
        assert_relative_eq!(new_cost, result.final_cost, epsilon = 1e-7);
    }
}
