use rand::{Rng, SeedableRng};

use crate::alignment::Sequences;
use crate::random::RandomGenerator;
use crate::tree::Tree;
use crate::Result;

/// A trait for building phylogenetic trees from a set of sequences.
pub trait TreeBuilder {
    fn build(
        self,
        seqs: &Sequences,
        rng: &mut RandomGenerator<impl Rng + SeedableRng>,
    ) -> Result<Tree>;
}
