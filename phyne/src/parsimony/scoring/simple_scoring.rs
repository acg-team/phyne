use std::fmt::Display;

use crate::alphabets::ParsimonySet;
use crate::parsimony::{GapCost, ParsimonyScoring};

#[derive(Clone, Debug, PartialEq)]
pub struct SimpleScoring {
    mismatch: f64,
    gap: GapCost,
}

impl SimpleScoring {
    pub fn new(mismatch: f64, gap: GapCost) -> SimpleScoring {
        SimpleScoring {
            mismatch,
            gap: gap * mismatch,
        }
    }
}

impl Display for SimpleScoring {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Simple parsimony scoring with mismatch: {}, {}",
            self.mismatch,
            self.gap * (1.0 / self.mismatch)
        )
    }
}

impl ParsimonyScoring for SimpleScoring {
    fn r#match(&self, _: f64, char_i: &u8, char_j: &u8) -> f64 {
        if char_i == char_j {
            0.0
        } else {
            self.mismatch
        }
    }

    fn min_match(&self, _: f64, i: &ParsimonySet, j: &ParsimonySet) -> f64 {
        if !((i & j).is_empty()) {
            0.0
        } else {
            self.mismatch
        }
    }

    fn gap_ext(&self, _: f64) -> f64 {
        self.gap.ext
    }

    fn gap_open(&self, _: f64) -> f64 {
        self.gap.open
    }

    fn avg(&self, _: f64) -> f64 {
        self.mismatch
    }
}
