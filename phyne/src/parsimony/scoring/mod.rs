use std::fmt::{Debug, Display};
use std::ops::Mul;

use dyn_clone::DynClone;

use crate::alphabets::ParsimonySet;

pub mod model_scoring;
pub use model_scoring::*;
pub mod simple_scoring;
pub use simple_scoring::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GapCost {
    open: f64,
    ext: f64,
}

impl Display for GapCost {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Gap cost multipliers [open: {}, ext: {}]",
            self.open, self.ext
        )
    }
}

impl Mul<f64> for GapCost {
    type Output = GapCost;

    fn mul(self, rhs: f64) -> GapCost {
        GapCost {
            open: self.open * rhs,
            ext: self.ext * rhs,
        }
    }
}

impl GapCost {
    pub fn new(open: f64, ext: f64) -> GapCost {
        GapCost { open, ext }
    }
}

pub trait ParsimonyScoring: Debug + Display + DynClone {
    fn r#match(&self, blen: f64, i: &u8, j: &u8) -> f64;
    fn min_match(&self, blen: f64, i: &ParsimonySet, j: &ParsimonySet) -> f64;
    fn gap_open(&self, blen: f64) -> f64;
    fn gap_ext(&self, blen: f64) -> f64;
    fn avg(&self, blen: f64) -> f64;
}

dyn_clone::clone_trait_object!(ParsimonyScoring);

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
