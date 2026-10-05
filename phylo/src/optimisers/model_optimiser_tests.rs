use std::fmt::Display;
use std::path::Path;

use approx::assert_relative_eq;

use crate::evolutionary_models::{EvoModel, FrequencyOptimisation};
use crate::frequencies;
use crate::likelihood::ModelSearchCost;
use crate::optimisers::{ModelOptimiser, StopCondition};
use crate::phylo_info::PhyloInfoBuilder as PIB;
use crate::pip_model::{PIPCostBuilder, PIPModel};
use crate::substitution_models::{
    dna_models::*, protein_models::*, FreqVector, QMatrix, QMatrixMaker, SubstModel,
    SubstitutionCostBuilder as SCB,
};
use crate::tkf_model::{TKF91CostBuilder, TKF92CostBuilder};

#[test]
fn likelihood_improves_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[0.25; 4], &[4.0]).unwrap();

    let c = SCB::new(model, info.clone()).build().unwrap();
    let unopt_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_cost > unopt_logl);

    let model = o.cost.model.clone();
    assert_relative_eq!(model.params()[0], 1.884815, epsilon = 1e-5);
    let c2 = SCB::new(model, info).build().unwrap();
    assert_relative_eq!(o.initial_cost, unopt_logl);
    assert_relative_eq!(o.final_cost, o.cost.cost());
    assert_relative_eq!(o.final_cost, c2.cost());
    assert_relative_eq!(o.final_cost, -4034.500803, epsilon = 1e-5);
}

#[test]
fn frequencies_unchanged_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[0.25; 4], &[4.0]).unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(initial_logl, o.costs[0]);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_ne!(o.cost.model.params(), model.params());
    assert_relative_eq!(o.cost.freqs(), &frequencies!(&[0.25; 4]));
    assert_relative_eq!(o.cost.freqs(), model.freqs());
}

#[test]
fn parameter_change_k80() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<K80>::new(&[0.25; 4], &[4.0]).unwrap();
    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(initial_logl, o.costs[0]);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert_ne!(o.cost.model.params(), model.params());
    assert_relative_eq!(o.cost.freqs(), &frequencies!(&[0.25; 4]));
    assert_relative_eq!(o.cost.freqs(), model.freqs());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn gtr_on_k80_data() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("K80/K80.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::new(&[0.25, 0.35, 0.3, 0.1], &[0.88, 0.03, 0.00001, 0.07, 0.02])
        .unwrap();
    let c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let o_gtr = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    let model = SubstModel::<K80>::new(&[0.25; 4], &[4.0]).unwrap();
    let c = SCB::new(model.clone(), info.clone()).build().unwrap();
    let o_k80 = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert!(o_gtr.final_cost >= o_k80.final_cost);
    assert_relative_eq!(
        o_gtr.cost.freqs(),
        &frequencies!(&[0.25; 4]),
        epsilon = 1e-1
    );
}

#[cfg(test)]
fn improved_logl_fixed_freqs_template<Q: QMatrix + Default>() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<Q>::default();

    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(initial_logl, o.costs[0]);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), model.params());
    assert_eq!(o.cost.freqs(), model.freqs());
}

#[test]
fn improved_logl_fixed_freqs() {
    improved_logl_fixed_freqs_template::<K80>();
    improved_logl_fixed_freqs_template::<HKY>();
    improved_logl_fixed_freqs_template::<TN93>();
    improved_logl_fixed_freqs_template::<GTR>();
}

#[cfg(test)]
fn improved_logl_empirical_freqs_template<Q: QMatrix + Default>() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<Q>::default();

    let c = SCB::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(initial_logl, o.costs[0]);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), model.params());
    assert_ne!(o.cost.freqs(), model.freqs());
    assert_eq!(o.cost.freqs(), &o.cost.empirical_freqs());
}

#[test]
fn improved_logl_empirical_freqs() {
    improved_logl_empirical_freqs_template::<HKY>();
    improved_logl_empirical_freqs_template::<TN93>();
    improved_logl_empirical_freqs_template::<GTR>();
}

#[cfg(test)]
fn improved_logl_empirical_freqs_pip_template<Q: QMatrix + QMatrixMaker + Default>() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = PIPModel::<Q>::with_default_substitution(&[1.4, 0.6]).unwrap();

    let c = PIPCostBuilder::new(model.clone(), info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();

    assert_eq!(initial_logl, o.initial_cost);
    assert_eq!(initial_logl, o.costs[0]);
    assert_eq!(o.cost.cost(), o.final_cost);
    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), model.params());
    assert_ne!(o.cost.freqs(), model.freqs());
    assert_eq!(
        o.cost.freqs().view((0, 0), (4, 1)),
        o.cost.empirical_freqs()
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn improved_logl_empirical_freqs_pip() {
    improved_logl_empirical_freqs_pip_template::<HKY>();
    improved_logl_empirical_freqs_pip_template::<TN93>();
    improved_logl_empirical_freqs_pip_template::<GTR>();
}

#[test]
fn gtr_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 0.031184397, 0.000100000, 0.077275972, 0.041508690],
    )
    .unwrap();
    let phyml_logl = SCB::new(phyml_model, info.clone()).build().unwrap().cost();
    assert_relative_eq!(phyml_logl, -3474.48083, epsilon = 1.0e-5);

    let paml_model = SubstModel::<GTR>::new(
        &[0.25318, 0.32894, 0.31196, 0.10592],
        &[0.88892, 0.03190, 0.00001, 0.07102, 0.02418],
    )
    .unwrap(); // Original input to paml
    let paml_logl = SCB::new(paml_model, info.clone()).build().unwrap().cost();
    assert!(phyml_logl > paml_logl);

    let model = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c = SCB::new(model, info.clone()).build().unwrap();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o.final_cost > phyml_logl);
    assert!(o.final_cost > paml_logl);

    let o2 = ModelOptimiser::new(o.cost, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert!(o2.final_cost >= o.final_cost);
    assert!(o2.iterations < 10);
}

#[test]
fn k80_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    // Optimized parameters from PhyML
    let phyml_model = SubstModel::<K80>::new(&[0.25; 4], &[19.432093]).unwrap();
    let phyml_logl = SCB::new(phyml_model.clone(), info.clone())
        .build()
        .unwrap()
        .cost();
    assert_relative_eq!(phyml_logl, -3629.2205979421, epsilon = 1.0e-5);

    let model = SubstModel::<K80>::new(&[0.25; 4], &[2.0]).unwrap();
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert!(o.final_cost > phyml_logl);

    let o2 = ModelOptimiser::new(o.cost.clone(), FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(o2.final_cost, o.final_cost, epsilon = 1e-10);
    assert!(o2.final_cost >= o.final_cost);
    assert!(o2.iterations < 10);
    assert_relative_eq!(o.cost.model.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_cost, phyml_logl, epsilon = 1e-6);
}

#[test]
fn hky_vs_phyml() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<HKY>::default();
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Empirical,
    )
    .run()
    .unwrap();

    let o2 = ModelOptimiser::new(o.cost.clone(), FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_relative_eq!(o2.final_cost, o.final_cost);
    assert!(o2.iterations < 10);

    // Optimized parameters from PhyML
    let phyml_model =
        SubstModel::<HKY>::new(&[0.24720, 0.35320, 0.29540, 0.10420], &[20.357397]).unwrap();
    let phyml_logl = SCB::new(phyml_model.clone(), info).build().unwrap().cost();
    assert_relative_eq!(phyml_logl, -3483.9223510041406, epsilon = 1.0e-5);

    assert!(o.final_cost >= phyml_logl);
    assert_relative_eq!(o.cost.freqs(), phyml_model.freqs());
    assert_relative_eq!(o.cost.model.params(), phyml_model.params(), epsilon = 1e-2);
    assert_relative_eq!(o.final_cost, phyml_logl, epsilon = 1e-5);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn frequencies_fixed_opt_gtr() {
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let model = SubstModel::<GTR>::new(&[0.25, 0.35, 0.3, 0.1], &[1.0; 5]).unwrap();
    let o = ModelOptimiser::new(
        SCB::new(model, info.clone()).build().unwrap(),
        FrequencyOptimisation::Fixed,
    )
    .run()
    .unwrap();
    assert_eq!(o.cost.freqs(), &frequencies!(&[0.25, 0.35, 0.3, 0.1]));
    assert_ne!(o.cost.model.params(), &[1.0; 5]);
    assert_eq!(o.final_cost, o.cost.cost());
}

#[cfg(test)]
fn frequencies_fixed_protein_template<Q: QMatrix + Default>() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::default();
    let c = SCB::new(model.clone(), info).build().unwrap();

    let initial_llik = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_cost);
    assert_eq!(initial_llik, o.final_cost);
    assert_eq!(model.freqs(), o.cost.freqs());
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
fn frequencies_fixed_protein() {
    frequencies_fixed_protein_template::<WAG>();
    frequencies_fixed_protein_template::<HIVB>();
    frequencies_fixed_protein_template::<BLOSUM>();
}

#[cfg(test)]
fn frequencies_empirical_protein_template<Q: QMatrix + Default>() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_protein1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    let model = SubstModel::<Q>::default();
    let c = SCB::new(model.clone(), info).build().unwrap();

    let initial_llik = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(initial_llik, o.initial_cost);
    assert_ne!(initial_llik, o.final_cost);
    assert_ne!(model.freqs(), o.cost.freqs());
}

#[test]
fn frequencies_empirical_protein() {
    frequencies_empirical_protein_template::<WAG>();
    frequencies_empirical_protein_template::<HIVB>();
    frequencies_empirical_protein_template::<BLOSUM>();
}

#[test]
fn arpip_example() {
    let fldr = Path::new("./data/pip/arpip/");
    let info = PIB::with_attrs(fldr.join("msa.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::with_default_substitution(&[0.1, 0.1]).unwrap();
    let c = PIPCostBuilder::new(pip_gtr.clone(), info.clone())
        .build()
        .unwrap();
    let initial_logl = c.cost();

    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(o.initial_cost, initial_logl);
    assert_relative_eq!(o.initial_cost, -212.3260420492571, epsilon = 1e-6); // value from the python script

    assert!(o.final_cost > initial_logl);
    assert_ne!(o.cost.model.params(), pip_gtr.params());
    assert_ne!(o.cost.freqs(), pip_gtr.freqs());
    assert_eq!(o.final_cost, o.cost.cost());

    let c = PIPCostBuilder::new(o.cost.model, info.clone())
        .build()
        .unwrap();
    assert_eq!(o.final_cost, c.cost());

    assert_relative_eq!(o.final_cost, -161.71033707196486, epsilon = 1e-6); // value from python script
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_propip_example() {
    let fldr = Path::new("./data/pip/propip/");
    let info = PIB::with_attrs(fldr.join("msa.initial.fasta"), fldr.join("tree.nwk"))
        .build()
        .unwrap();

    let pip_gtr = PIPModel::<GTR>::with_default_substitution(&[14.142_1, 0.1414]).unwrap();
    let c = PIPCostBuilder::new(pip_gtr.clone(), info).build().unwrap();
    let initial_logl = c.cost();

    assert_relative_eq!(initial_logl, -1241.99424826303, epsilon = 1e-8); // value from the python script
    assert_relative_eq!(initial_logl, -1241.9944955187807, epsilon = 1e-3); // value from ProPIP
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_eq!(o.initial_cost, initial_logl);
    assert!(o.final_cost > initial_logl);
    assert_relative_eq!(o.final_cost, -1081.7242569614295, epsilon = 1e-8); // value from the python script
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_vs_python_no_gaps() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("Huelsenbeck_example_long_DNA.fasta"),
        fldr.join("Huelsenbeck_example.newick"),
    )
    .build()
    .unwrap();

    let pip_hky = PIPModel::<JC69>::with_default_substitution(&[1.2, 0.45]).unwrap();

    let c = PIPCostBuilder::new(pip_hky, info).build().unwrap();
    assert_relative_eq!(c.cost(), -361.18634412281443, epsilon = 1e-7); // value from the python script
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_ne!(o.cost.param(0), 1.2);
    assert_ne!(o.cost.param(1), 0.45);
    assert_ne!(o.cost.param(2), 1.0);
    assert!(o.final_cost > -361.18634412281443);
    assert_relative_eq!(o.final_cost, -227.0848511231626, epsilon = 1e-7); // value from the python script
    assert_eq!(o.final_cost, o.cost.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_gtr_optimisation() {
    // Check that pip parameter optimisation produces expected results
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();

    let initial_logl = c.cost();
    let pip_o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    let c = PIPCostBuilder::new(pip_o.cost.model.clone(), info.clone())
        .build()
        .unwrap();

    assert_eq!(pip_o.final_cost, c.cost());

    assert_relative_eq!(initial_logl, -9988.840775519875, epsilon = 1e-5); // value from the python script
    assert_relative_eq!(pip_o.initial_cost, initial_logl);
    assert!(pip_o.final_cost > initial_logl);
    // On Intel chips epsilon = 1e-5 was sufficient, but not on ARM. See https://github.com/acg-team/rust-phylo/issues/80
    assert_relative_eq!(pip_o.final_cost, -3481.829388139959, epsilon = 1e-4); // value from the python script
    assert_eq!(pip_o.final_cost, pip_o.cost.cost());
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_gtr_vs_gtr_params() {
    // Compare pip gtr parameter optimisation vs the original gtr model
    let fldr = Path::new("./data/sim");
    let info = PIB::with_attrs(fldr.join("GTR/gtr.fasta"), fldr.join("tree.newick"))
        .build()
        .unwrap();
    let pip_gtr = PIPModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c = PIPCostBuilder::new(pip_gtr, info.clone()).build().unwrap();
    let pip_o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();

    let gtr = SubstModel::<GTR>::new(
        &[0.24720, 0.35320, 0.29540, 0.10420],
        &[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    .unwrap();
    let c = SCB::new(gtr, info).build().unwrap();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    assert_relative_eq!(
        pip_o.cost.model.params()[2..],
        o.cost.model.params(),
        epsilon = 1e-2
    );
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn pip_protein_example() {
    let fldr = Path::new("./data/phyml_protein_example/");
    let info = PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("example_tree.newick"))
        .build()
        .unwrap();
    let pip = PIPModel::<WAG>::with_default_substitution(&[2.0, 0.1]).unwrap();
    let c = PIPCostBuilder::new(pip, info).build().unwrap();
    let initial_logl = c.cost();
    let o = ModelOptimiser::new(c, FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert!(o.final_cost > initial_logl);
    assert_relative_eq!(o.initial_cost, initial_logl);
    assert_ne!(o.cost.param(0), 2.0);
    assert_ne!(o.cost.param(1), 0.1);
    assert_eq!(o.cost.cost(), o.final_cost);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn stop_condition_epsilon() {
    let epsilon = 0.005;
    let fldr = Path::new("./data/phyml_protein_example/");
    let info = PIB::with_attrs(fldr.join("seqs.fasta"), fldr.join("example_tree.newick"))
        .build()
        .unwrap();
    let pip = PIPModel::<WAG>::with_default_substitution(&[2.0, 0.1]).unwrap();
    let c = PIPCostBuilder::new(pip, info).build().unwrap();
    let initial_logl = c.cost();
    let result = ModelOptimiser::with_stop_condition(
        c,
        FrequencyOptimisation::Empirical,
        StopCondition::epsilon(epsilon),
    )
    .run()
    .unwrap();

    assert!(result.final_cost > initial_logl);
    assert_relative_eq!(result.initial_cost, initial_logl);
    assert_ne!(result.cost.param(0), 2.0);
    assert_ne!(result.cost.param(1), 0.1);
    assert_eq!(result.cost.cost(), result.final_cost);
    let mut costs = result.costs;
    assert!(costs.pop().unwrap() - costs.pop().unwrap() < epsilon);
}

#[cfg(test)]
fn tkf_model_opti_template<C: ModelSearchCost + Clone + Display>(c: C) {
    let initial_llik = c.cost();
    let o = ModelOptimiser::new(c.clone(), FrequencyOptimisation::Fixed)
        .run()
        .unwrap();
    let intermediate_cost = o.final_cost;
    assert_eq!(initial_llik, o.initial_cost);
    assert_eq!(o.final_cost, o.cost.cost());
    assert!(o.initial_cost < o.final_cost);
    assert_eq!(c.freqs(), o.cost.freqs());
    for param in 0..c.param_count() {
        assert_ne!(c.param(param), o.cost.param(param));
        let valid_range = o.cost.param_range(param);
        assert!(valid_range.0 <= o.cost.param(param) && o.cost.param(param) <= valid_range.1);
    }

    let o = ModelOptimiser::new(o.cost.clone(), FrequencyOptimisation::Empirical)
        .run()
        .unwrap();
    assert_eq!(intermediate_cost, o.initial_cost);
    assert_eq!(o.final_cost, o.cost.cost());
    assert!(o.initial_cost < o.final_cost);
    assert_eq!(o.cost.freqs(), &o.cost.empirical_freqs());
    for param in 0..c.param_count() {
        assert_ne!(c.param(param), o.cost.param(param));
        let valid_range = o.cost.param_range(param);
        assert!(valid_range.0 <= o.cost.param(param) && o.cost.param(param) <= valid_range.1);
    }
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn tkf91_model_opti() {
    let fldr = Path::new("./data/pip/arpip/");
    let info = PIB::with_attrs(fldr.join("msa.fasta"), fldr.join("tree.nwk"))
        .build_with_ancestors()
        .unwrap();
    let subst_model = SubstModel::<HKY>::default();
    let tkf91 = TKF91CostBuilder::new(&[0.8, 1.0], subst_model.clone(), info.clone())
        .build()
        .unwrap();
    tkf_model_opti_template(tkf91);
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn tkf92_model_opti() {
    let fldr = Path::new("./data/pip/arpip/");
    let info = PIB::with_attrs(fldr.join("msa.fasta"), fldr.join("tree.nwk"))
        .build_with_ancestors()
        .unwrap();
    let subst_model = SubstModel::<HKY>::default();
    let tkf92 = TKF92CostBuilder::new(&[0.8, 1.0, 0.2], subst_model, info)
        .build()
        .unwrap();
    tkf_model_opti_template(tkf92);
}
