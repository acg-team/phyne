use std::fmt::Display;

use approx::relative_eq;
use log::warn;

use crate::alphabets::{Alphabet, AMINOACID_INDEX};
use crate::likelihood::{ParamRange, PARAM_RANGE_DUMMY};
use crate::substitution_models::{FreqVector, QMatrix, QMatrixMaker, SubstMatrix};
use crate::{bail, Result};

pub(crate) mod protein_generics;
pub(crate) use protein_generics::*;

const PROTEIN_N: usize = 20;

pub fn make_exchangeability(lower_triangle: &ProteinExchLowerTriangle) -> ProteinExch {
    let mut exch = [0.0; 400];
    let mut idx = 0;
    for i in 1..20 {
        for j in 0..i {
            exch[i * 20 + j] = lower_triangle[idx];
            exch[j * 20 + i] = lower_triangle[idx];
            idx += 1;
        }
    }
    exch
}

fn make_protein_q(exchangeability: &SubstMatrix, freqs: &FreqVector) -> SubstMatrix {
    let mut q = exchangeability * SubstMatrix::from_diagonal(freqs);
    for i in 0..PROTEIN_N {
        q[(i, i)] = -q.row(i).sum();
    }
    let scaler = -1.0 / q.diagonal().component_mul(freqs).sum();
    q.scale_mut(scaler);
    q
}

fn validate_protein_frequencies(freqs: &FreqVector) -> Result<()> {
    if freqs.len() < PROTEIN_N {
        bail!(
            SubstitutionModel,
            "too few frequencies provided for protein model"
        );
    } else if freqs.len() > PROTEIN_N {
        bail!(
            SubstitutionModel,
            "too many frequencies provided for protein model"
        );
    } else if freqs.into_iter().any(|x| *x < 0.0) {
        bail!(
            SubstitutionModel,
            "one or more negative frequencies provided for protein model"
        );
    } else if !relative_eq!(freqs.into_iter().sum::<f64>().abs(), 1.0) {
        bail!(
            SubstitutionModel,
            "frequencies for protein model do not sum to 1.0"
        );
    }
    Ok(())
}

macro_rules! define_protein_model {
    ($name:ident, $pi:ident, $exch:expr) => {
        #[derive(Clone, Debug, PartialEq)]
        #[allow(clippy::upper_case_acronyms)]
        pub struct $name {
            freqs: FreqVector,
            q: SubstMatrix,
            exchangeability: SubstMatrix,
        }
        impl QMatrixMaker for $name {
            fn create(freqs: &[f64], _: &[f64]) -> $name {
                let freqs = FreqVector::from_column_slice(freqs);
                let freqs = if let Err(err) = validate_protein_frequencies(&freqs) {
                    warn!("Invalid protein frequencies: {}", err);
                    warn!("Falling back to default protein frequencies");
                    FreqVector::from_column_slice(&$pi)
                } else {
                    freqs
                };
                let exchangeability = SubstMatrix::from_row_slice(PROTEIN_N, PROTEIN_N, &$exch);
                let q = make_protein_q(&exchangeability, &freqs);
                $name {
                    freqs,
                    q,
                    exchangeability,
                }
            }
        }
        impl QMatrix for $name {
            fn q(&self) -> &SubstMatrix {
                &self.q
            }

            fn freqs(&self) -> &FreqVector {
                &self.freqs
            }

            fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
                validate_protein_frequencies(&freqs)?;
                self.freqs = freqs;
                self.q = make_protein_q(&self.exchangeability, &self.freqs);
                Ok(())
            }

            fn param_count(&self) -> usize {
                0
            }

            fn param_range(&self, _: usize) -> ParamRange {
                PARAM_RANGE_DUMMY
            }

            fn set_param(&mut self, _: usize, _: f64) {}

            fn params(&self) -> &[f64] {
                &[]
            }

            fn n(&self) -> usize {
                PROTEIN_N
            }

            fn rate(&self, i: u8, j: u8) -> f64 {
                self.q[(AMINOACID_INDEX[i as usize], AMINOACID_INDEX[j as usize])]
            }

            fn alphabet() -> &'static Alphabet {
                Alphabet::protein()
            }
        }

        impl Display for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                write!(
                    f,
                    "{} with [pi = {:?}]",
                    stringify!($name),
                    self.freqs().as_slice()
                )
            }
        }
    };
}

define_protein_model!(WAG, WAG_PI, WAG_EXCH);
define_protein_model!(HIVB, HIVB_PI, make_exchangeability(&HIVB_EXCH_LOWER_TRIAG));
define_protein_model!(BLOSUM, BLOSUM_PI, BLOSUM_EXCH);

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use rstest::rstest;

    use assert_matches::assert_matches;

    use crate::{frequencies, Error};

    use super::*;

    #[rstest]
    #[case::negative_sum_1(&[
        -10.0, 10.0, 0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    ])]
    #[case::all_negative(&[-0.1; 20])]
    #[case::one_negative_sum_1(&[-0.1, 0.5, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])]
    fn negative_frequencies(#[case] freqs: &[f64]) {
        assert_matches!(validate_protein_frequencies(&frequencies!(freqs)), Err(Error::SubstitutionModel(msg)) if msg.contains("negative") & msg.contains("protein"));
    }

    #[rstest]
    #[case::too_few_sum_1(&[1.0/19.0; 19])]
    #[case::valid_dna(&[1.0/4.0; 4])]
    #[case::too_few(&[0.1, 0.4, 0.0])]
    #[case::empty(&[])]
    fn too_few_frequencies(#[case] freqs: &[f64]) {
        assert_matches!(validate_protein_frequencies(&frequencies!(freqs)), Err(Error::SubstitutionModel(msg)) if msg.contains("too few") & msg.contains("protein"));
    }

    #[rstest]
    #[case::one_too_many_sum_1(&[1.0/21.0; 21])]
    #[case::too_many(&[0.1; 30])]
    fn too_many_frequencies(#[case] freqs: &[f64]) {
        assert_matches!(validate_protein_frequencies(&frequencies!(freqs)), Err(Error::SubstitutionModel(msg)) if msg.contains("too many") & msg.contains("protein"));
    }

    #[rstest]
    #[case::sum_below_1(&[0.001; 20])]
    #[case::sum_above_1(&[0.1; 20])]
    #[case::sum_above_1_large(&[1.1; 20])]

    fn frequencies_dont_sum_to_1(#[case] freqs: &[f64]) {
        assert_matches!(validate_protein_frequencies(&frequencies!(freqs)), Err(Error::SubstitutionModel(msg)) if msg.contains("do not sum to 1") & msg.contains("protein"));
    }
}
