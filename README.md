# Phyne

A high-performance Rust crate for indel-aware phylogenetic analysis and multiple sequence alignment under maximum likelihood and parsimony optimality criteria.

<!-- [![Crates.io](https://img.shields.io/crates/v/phyne.svg)](https://crates.io/crates/phyne)
[![Documentation](https://docs.rs/phyne/badge.svg)](https://docs.rs/phyne) -->
[![Licence](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue.svg)](https://github.com/acg-team/phyne#license) [![CI](https://github.com/acg-team/phyne/actions/workflows/test_stable.yml/badge.svg)](https://github.com/acg-team/phyne/actions) [![codecov](https://codecov.io/gh/acg-team/phyne/branch/develop/graph/badge.svg)](https://codecov.io/gh/acg-team/phyne)

[Current Functionality](#current-functionality) • [Getting Started](#getting-started) • [Crate Features](#crate-features) • [Roadmap](#roadmap) • [Contributing](#contributing) • [Related Projects](#related-projects) • [Support](#support) • [Citation](#citation) • [Licence and Attributions](#licence-and-attributions)

## Current Functionality

- **Maximum Likelihood Phylogenetic Analysis**: Efficient implementation of phylogenetic tree inference using SPR or NNI moves using likelihood or parsimony cost functions;
- **Multiple Sequence Alignment (MSA)**: Support for Multiple Sequence Alignment using the [IndelMaP algorithm]( https://academic.oup.com/mbe/article/41/7/msae109/7688856 );
- **Sequence Evolution Models**: Support for various DNA (JC69, K80, TN93, HKY, GTR) and protein (WAG, HIVB, BLOSUM62) substitution models as well as the [Poisson Indel Process (PIP) model]( https://www.pnas.org/doi/10.1073/pnas.1220450110 ) and the [TKF91](https://link.springer.com/article/10.1007/BF02193625)/[TKF92](https://link.springer.com/article/10.1007/BF00163848) models (the TKF models are only compatible with NNI moves, see the documentation of the `Compatible` trait);
- **High Performance**: Optimised tree search with optional parallel processing capabilities.

## Getting Started

**Note**: This crate is not yet published on crates.io. To use it directly from GitHub, add this to your `Cargo.toml`:

```toml
[dependencies]
phyne = { git = "https://github.com/acg-team/phyne", package = "phyne" }
```

Once published on crates.io, you'll be able to use:

```toml
[dependencies]
phyne = "0.1.0"
```

**Minimum Supported Rust Version**: 1.88.0

MSRV detected using [`cargo-msrv`]( https://github.com/foresterre/cargo-msrv ).

### Example

<!-- WARNING: If you change the code below, please also update the test "phyne/tests/topology_optimiser.rs:example_main_from_readme" accordingly!) -->

```rust
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
```

## Crate Features

This crate supports several optional features:

- `par-regraft`: Enable parallel regrafting operations using Rayon;
- `par-regraft-chunk`: Enable chunked parallel regrafting;
- `par-regraft-manual`: Enable manual parallel regrafting control;

Test related features:

- `precomputed-test-results`: Speed up test runs with precomputed results (for local development);
- `recompute-brute-force-asr`: Force re-computation of brute-force ancestral sequence re-estimation in tests instead of loading precomputed results;
- `multithread-brute-force-asr`: Parallelise brute-force ancestral sequence re-estimation computation in tests.

Enable features in your Cargo.toml:

```toml
[dependencies]
phyne = { git = "https://github.com/acg-team/phyne", package = "phyne", features = ["par-regraft"] }
```

<!--
## Documentation
Full documentation is available at docs.rs/phyne.
-->

<!--
## Performance benchmarks
Will add Luca's benchmarks when possible.
-->

## Roadmap

This crate is new and in active development at the moment. The basic existing functionality is mentioned above, but the following features are being currently implemented or planned:

- Simultaneous tree and alignment estimation under the [PIP model]( https://www.pnas.org/doi/10.1073/pnas.1220450110 );
- Extension to the PIP model that includes long insertions (manuscript in preparation);
- Ancestral state reconstruction: using PIP ([ARPIP]( https://pubmed.ncbi.nlm.nih.gov/35866991/ )), TKF92 and [IndelMaP]( https://academic.oup.com/mbe/article/41/7/msae109/7688856 );
- Randomised starting trees for tree inference;
- Generalisation of the tree structure for easier use in other crates.

Other minor features/improvements are documented on the [GitHub issues page]( https://github.com/acg-team/phyne/issues?q=is%3Aissue%20state%3Aopen%20label%3Aenhancement ).

## Contributing

This is a new library that is currently in active development. Contributions are highly welcome!

**API Stability**: As this crate is in active development, the API may change between versions until we reach 1.0. We'll follow semantic versioning and document breaking changes in release notes.

### New Contributors

Please read our [contributor guide]( CONTRIBUTING.md )!

### Contributors:

#### Current
- Jūlija Pečerska ([GitHub]( https://github.com/junniest ), [email]( mailto:julija.pecerska@zhaw.ch ));
- Mattes Mrzik ([GitHub]( https://github.com/MattesMrzik ), [email]( mailto:mattes.mrzik@zhaw.ch ));
#### Past
- Dmitrii Iartsev ([GitHub]( https://github.com/jarcev ));
- Merlin Maggi ([GitHub]( https://github.com/merlinio2000 ));
- Luca Müller ([GitHub]( https://github.com/lucasperception ));
- Kai Davidson ([GitHub]( https://github.com/kalxed )).

## Related Projects

- [JATI (joint alignment tree inference)]( https://github.com/acg-team/JATI )

## Support

For questions, bug reports, or feature requests, please go to the [`phyne` discussion page]( https://github.com/acg-team/phyne/discussions ) and/or open an issue on [GitHub]( https://github.com/acg-team/phyne/issues ).

## Citation

If you use this library in your research, please consider citing:

```bibtex
@software{phyne_rust,
  title = {This is Phyne: a Rust crate for indel-aware phylogenetics},
  author = {Pečerska, Jūlija and Mrzik, Mattes and Gil, Manuel and Anisimova, Maria},
  url = {https://github.com/acg-team/phyne},
  year = {2026}
}
```

## Licence and Attributions

### Licence

This project is licensed under either of
- Apache Licence, Version 2.0 ([LICENSE-APACHE]( LICENSE-APACHE ) or [www.apache.org/licenses/LICENSE-2.0]( http://www.apache.org/licenses/LICENSE-2.0 )), or
- MIT Licence ([LICENSE-MIT]( LICENSE-MIT ) or [opensource.org/licenses/MIT]( http://opensource.org/licenses/MIT ))

at your option.

### Benchmarking Datasets

Datasets for benchmarking were taken from:
- Zhou, Xiaofan (2017). Single-gene alignments. figshare. Dataset. [Link]( https://doi.org/10.6084/m9.figshare.5477749.v1 );
- Zhou, Xiaofan (2017). Supermatrices. figshare. Dataset. [Link]( https://doi.org/10.6084/m9.figshare.5477746.v1 ).

The datasets are licensed under [CC BY 4.0]( https://creativecommons.org/licenses/by/4.0/ ).

The datasets were modified by normalising invalid/unrecognised sequence characters since the exact sequences are less relevant
for pure performance measurements.
