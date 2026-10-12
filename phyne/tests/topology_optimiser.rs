// Warning: If you change this test, please also update the example in the README.md file
// accordingly!
#[test]
fn example_main_from_readme() {
    use phyne::likelihood::TreeSearchCost;
    use phyne::optimisers::{SprOptimiser, TopologyOptimiser};
    use phyne::phylo_info::PhyloInfoBuilder;
    use phyne::random::FakeGenerator;
    use phyne::substitution_models::{SubstModel, SubstitutionCostBuilder, K80};
    use phyne::Result;

    fn main() -> Result<()> {
        // Note: This example uses test data from the repository
        let info = PhyloInfoBuilder::new("./examples/data/K80.fasta").build()?;
        let k80 = SubstModel::<K80>::default();
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
