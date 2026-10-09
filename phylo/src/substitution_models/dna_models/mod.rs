use std::fmt::Display;

use approx::relative_eq;

use crate::alphabets::{Alphabet, NUCLEOTIDE_INDEX};
use crate::frequencies;
use crate::likelihood::{ParamRange, PARAM_RANGE_DUMMY, PARAM_RANGE_POSITIVE};
use crate::substitution_models::{FreqVector, QMatrix, QMatrixMaker, SubstMatrix};
use crate::{bail, Result};

const DNA_N: usize = 4;
const EQUAL_FREQS: [f64; DNA_N] = [0.25, 0.25, 0.25, 0.25];

pub(super) const JC69_PARAM_N: usize = 0;
pub(super) const K80_PARAM_N: usize = 1;
pub(super) const HKY_PARAM_N: usize = 1;
pub(super) const TN93_PARAM_N: usize = 2;
pub(super) const GTR_PARAM_N: usize = 5;

// Default transition/transversion ratio for K80 and HKY models
pub(super) const DEFAULT_TS_TV_RATIO: f64 = 2.0;
pub(super) const DEFAULT_GTR_RATES: [f64; GTR_PARAM_N] = [1.0, 1.0, 1.0, 1.0, 1.0];
pub(super) const DEFAULT_TN93_RATES: [f64; TN93_PARAM_N] = [1.0, 1.0];

fn validate_dna_frequencies(freqs: &FreqVector) -> Result<()> {
    if freqs.len() != DNA_N {
        bail!(SubstitutionModel, FrequencyCount, "DNA", DNA_N, freqs.len());
    } else if freqs.iter().any(|x| *x < 0.0) {
        bail!(SubstitutionModel, NegativeFrequency, "DNA");
    } else if !relative_eq!(freqs.iter().sum::<f64>().abs(), 1.0, epsilon = 1e-10) {
        bail!(SubstitutionModel, FrequencySum, "DNA");
    } else if freqs.iter().any(|x| *x == 1.0) {
        bail!(
            SubstitutionModel,
            DegenerateFrequencies,
            "DNA",
            freqs.iter().position(|x| *x == 1.0).unwrap()
        );
    }
    Ok(())
}

#[derive(Clone, Debug, PartialEq)]
pub struct JC69 {
    freqs: FreqVector,
    q: SubstMatrix,
}

impl Default for JC69 {
    fn default() -> Self {
        let r = 1.0 / 3.0;
        let q = SubstMatrix::from_row_slice(
            DNA_N,
            DNA_N,
            &[-1.0, r, r, r, r, -1.0, r, r, r, r, -1.0, r, r, r, r, -1.0],
        );
        JC69 {
            freqs: frequencies!(&EQUAL_FREQS),
            q,
        }
    }
}

impl QMatrixMaker for JC69 {
    fn create(freqs: &[f64], params: &[f64]) -> Result<JC69> {
        let freqs = frequencies!(freqs);
        validate_dna_frequencies(&freqs)?;

        if freqs != frequencies!(&EQUAL_FREQS) {
            bail!(SubstitutionModel, UnequalFrequencies, "JC69");
        }

        if !params.is_empty() {
            bail!(
                SubstitutionModel,
                ParameterCount,
                "JC69",
                Self::param_count(),
                params.len()
            );
        }

        Ok(JC69::default())
    }
}

impl QMatrix for JC69 {
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(NUCLEOTIDE_INDEX[i as usize], NUCLEOTIDE_INDEX[j as usize])]
    }
    fn param_count() -> usize {
        JC69_PARAM_N
    }
    fn params(&self) -> &[f64] {
        &[]
    }
    fn set_param(&mut self, _: usize, _: f64) {}
    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_DUMMY
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        validate_dna_frequencies(&freqs)?;
        if freqs != frequencies!(&EQUAL_FREQS) {
            bail!(SubstitutionModel, UnequalFrequencies, "JC69");
        }
        Ok(())
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn alphabet() -> &'static Alphabet {
        Alphabet::dna()
    }
}

impl Display for JC69 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "JC69")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct K80 {
    freqs: FreqVector,
    q: SubstMatrix,
    kappa: Vec<f64>,
}

impl Default for K80 {
    fn default() -> Self {
        let kappa = DEFAULT_TS_TV_RATIO;
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        k80_q(&mut q, kappa);
        K80 {
            freqs: frequencies!(&EQUAL_FREQS),
            q,
            kappa: vec![kappa],
        }
    }
}

impl QMatrixMaker for K80 {
    fn create(freqs: &[f64], params: &[f64]) -> Result<K80> {
        let freqs = frequencies!(freqs);
        validate_dna_frequencies(&freqs)?;

        if freqs != frequencies!(&EQUAL_FREQS) {
            bail!(SubstitutionModel, UnequalFrequencies, "K80");
        }

        let kappa = if params.len() != K80::param_count() {
            bail!(
                SubstitutionModel,
                ParameterCount,
                "K80",
                K80::param_count(),
                params.len()
            );
        } else {
            params[0]
        };

        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        k80_q(&mut q, kappa);
        Ok(K80 {
            freqs: frequencies!(&EQUAL_FREQS),
            q,
            kappa: vec![kappa],
        })
    }
}

impl QMatrix for K80 {
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(NUCLEOTIDE_INDEX[i as usize], NUCLEOTIDE_INDEX[j as usize])]
    }
    fn param_count() -> usize {
        K80_PARAM_N
    }
    fn params(&self) -> &[f64] {
        &self.kappa
    }
    fn set_param(&mut self, _: usize, value: f64) {
        self.kappa[0] = value;
        k80_q(&mut self.q, value);
    }
    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_POSITIVE
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        validate_dna_frequencies(&freqs)?;
        if freqs != frequencies!(&EQUAL_FREQS) {
            bail!(SubstitutionModel, UnequalFrequencies, "K80");
        }
        Ok(())
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn alphabet() -> &'static Alphabet {
        Alphabet::dna()
    }
}

impl Display for K80 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "K80 with [kappa = {:.5}]", self.kappa[0])
    }
}

fn k80_q(q: &mut SubstMatrix, k: f64) {
    let scaler = 1.0 / (k * 0.25 + 0.5);
    q[(0, 0)] = -(k * 0.25 + 0.5);
    q[(0, 1)] = k * 0.25;
    q[(0, 2)] = 0.25;
    q[(0, 3)] = 0.25;

    q[(1, 0)] = k * 0.25;
    q[(1, 1)] = -(k * 0.25 + 0.5);
    q[(1, 2)] = 0.25;
    q[(1, 3)] = 0.25;

    q[(2, 0)] = 0.25;
    q[(2, 1)] = 0.25;
    q[(2, 2)] = -(0.5 + k * 0.25);
    q[(2, 3)] = k * 0.25;

    q[(3, 0)] = 0.25;
    q[(3, 1)] = 0.25;
    q[(3, 2)] = k * 0.25;
    q[(3, 3)] = -(0.5 + k * 0.25);
    q.scale_mut(scaler);
}

#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct HKY {
    freqs: FreqVector,
    q: SubstMatrix,
    kappa: Vec<f64>,
}

impl Default for HKY {
    fn default() -> Self {
        let kappa = DEFAULT_TS_TV_RATIO;
        let freqs = frequencies!(&EQUAL_FREQS);
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        hky_q(&mut q, &freqs, kappa);
        HKY {
            freqs,
            q,
            kappa: vec![kappa; Self::param_count()],
        }
    }
}

impl QMatrixMaker for HKY {
    fn create(freqs: &[f64], params: &[f64]) -> Result<HKY> {
        let freqs = frequencies!(freqs);
        validate_dna_frequencies(&freqs)?;

        let kappa = if params.len() != HKY::param_count() {
            bail!(
                SubstitutionModel,
                ParameterCount,
                "HKY",
                HKY::param_count(),
                params.len()
            );
        } else {
            params[0]
        };

        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        hky_q(&mut q, &freqs, kappa);
        Ok(HKY {
            freqs,
            q,
            kappa: vec![kappa; Self::param_count()],
        })
    }
}

impl QMatrix for HKY {
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(NUCLEOTIDE_INDEX[i as usize], NUCLEOTIDE_INDEX[j as usize])]
    }
    fn param_count() -> usize {
        HKY_PARAM_N
    }
    fn params(&self) -> &[f64] {
        &self.kappa
    }
    fn set_param(&mut self, _: usize, value: f64) {
        self.kappa[0] = value;
        hky_q(&mut self.q, &self.freqs, self.kappa[0])
    }
    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_POSITIVE
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        validate_dna_frequencies(&freqs)?;
        self.freqs = freqs;
        hky_q(&mut self.q, &self.freqs, self.kappa[0]);
        Ok(())
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn alphabet() -> &'static Alphabet {
        Alphabet::dna()
    }
}

fn hky_q(q: &mut SubstMatrix, pi: &FreqVector, k: f64) {
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let scaler = 1.0
        / ((k * fc + (fa + fg)) * ft
            + (k * ft + (fa + fg)) * fc
            + ((ft + fc) + k * fg) * fa
            + ((ft + fc) + k * fa) * fg);
    q[(0, 0)] = -(k * fc + (fa + fg));
    q[(0, 1)] = k * fc;
    q[(0, 2)] = fa;
    q[(0, 3)] = fg;

    q[(1, 0)] = k * ft;
    q[(1, 1)] = -(k * ft + (fa + fg));
    q[(1, 2)] = fa;
    q[(1, 3)] = fg;

    q[(2, 0)] = ft;
    q[(2, 1)] = fc;
    q[(2, 2)] = -((ft + fc) + k * fg);
    q[(2, 3)] = k * fg;

    q[(3, 0)] = ft;
    q[(3, 1)] = fc;
    q[(3, 2)] = k * fa;
    q[(3, 3)] = -((ft + fc) + k * fa);
    q.scale_mut(scaler);
}

impl Display for HKY {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "HKY with [kappa = {:.5}, freqs = {}]",
            self.kappa[0], self.freqs
        )
    }
}

/// Tamura-Nei 1993 (TN93) DNA substitution model.
///
/// This model allows for different rates of two different types of transitions (A <-> G and C <-> T).
/// Transversion rates are assumed to be equal and fixed to 1.0 to simplify parameter estimation as
/// the substitution matrix is normalised so that the average substitution rate is 1.0.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct TN93 {
    freqs: FreqVector,
    pub(crate) q: SubstMatrix,
    params: Vec<f64>,
}

impl Default for TN93 {
    fn default() -> Self {
        let params = DEFAULT_TN93_RATES.to_vec();
        let freqs = frequencies!(&EQUAL_FREQS);
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        tn93_q(&mut q, &freqs, &params);
        TN93 { freqs, q, params }
    }
}

impl QMatrixMaker for TN93 {
    fn create(freqs: &[f64], params: &[f64]) -> Result<TN93> {
        let freqs = frequencies!(freqs);
        validate_dna_frequencies(&freqs)?;

        if params.len() != Self::param_count() {
            bail!(
                SubstitutionModel,
                ParameterCount,
                "TN93",
                Self::param_count(),
                params.len()
            );
        }

        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        tn93_q(&mut q, &freqs, params);
        Ok(TN93 {
            freqs,
            q,
            params: params.to_vec(),
        })
    }
}

impl QMatrix for TN93 {
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(NUCLEOTIDE_INDEX[i as usize], NUCLEOTIDE_INDEX[j as usize])]
    }
    fn param_count() -> usize {
        TN93_PARAM_N
    }
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.params[param] = value;
        tn93_q(&mut self.q, &self.freqs, &self.params)
    }
    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_POSITIVE
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        validate_dna_frequencies(&freqs)?;
        self.freqs = freqs;
        tn93_q(&mut self.q, &self.freqs, &self.params);
        Ok(())
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn alphabet() -> &'static Alphabet {
        Alphabet::dna()
    }
}

fn tn93_q(q: &mut SubstMatrix, pi: &FreqVector, params: &[f64]) {
    // beta (transversion rate) is fixed to 1.0
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let a1 = params[0];
    let a2 = params[1];

    let scaler = 1.0
        / ((a1 * fc + fa + fg) * ft
            + (a1 * ft + fa + fg) * fc
            + (ft + fc + a2 * fg) * fa
            + (ft + fc + a2 * fa) * fg);

    q[(0, 0)] = -(a1 * fc + fa + fg);
    q[(0, 1)] = a1 * fc;
    q[(0, 2)] = fa;
    q[(0, 3)] = fg;

    q[(1, 0)] = a1 * ft;
    q[(1, 1)] = -(a1 * ft + fa + fg);
    q[(1, 2)] = fa;
    q[(1, 3)] = fg;

    q[(2, 0)] = ft;
    q[(2, 1)] = fc;
    q[(2, 2)] = -(ft + fc + a2 * fg);
    q[(2, 3)] = a2 * fg;

    q[(3, 0)] = ft;
    q[(3, 1)] = fc;
    q[(3, 2)] = a2 * fa;
    q[(3, 3)] = -(ft + fc + a2 * fa);

    q.scale_mut(scaler);
}

impl Display for TN93 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "TN93 with [kappa1 = {:.5}, kappa2 = {:.5}, freqs = {}]",
            self.params[0], self.params[1], self.freqs
        )
    }
}

/// GTR (General Time Reversible) DNA substitution model.
///
/// This model allows for different rates of all possible nucleotide substitutions.
/// The rate of A <-> G transition is fixed to 1.0 to simplify parameter estimation as
/// the substitution matrix is normalised so that the average substitution rate is 1.0.
#[derive(Clone, Debug, PartialEq)]
#[allow(clippy::upper_case_acronyms)]
pub struct GTR {
    freqs: FreqVector,
    q: SubstMatrix,
    params: Vec<f64>,
}

impl Default for GTR {
    fn default() -> Self {
        let params = DEFAULT_GTR_RATES.to_vec();
        let freqs = frequencies!(&EQUAL_FREQS);
        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        gtr_q(&mut q, &freqs, &params);
        GTR { freqs, q, params }
    }
}

impl QMatrixMaker for GTR {
    fn create(freqs: &[f64], params: &[f64]) -> Result<GTR> {
        let freqs = frequencies!(freqs);
        validate_dna_frequencies(&freqs)?;

        if params.len() != Self::param_count() {
            bail!(
                SubstitutionModel,
                ParameterCount,
                "GTR",
                Self::param_count(),
                params.len()
            );
        }

        let mut q = SubstMatrix::zeros(DNA_N, DNA_N);
        gtr_q(&mut q, &freqs, params);
        Ok(GTR {
            freqs,
            q,
            params: params.to_vec(),
        })
    }
}

impl QMatrix for GTR {
    fn q(&self) -> &SubstMatrix {
        &self.q
    }
    fn rate(&self, i: u8, j: u8) -> f64 {
        self.q[(NUCLEOTIDE_INDEX[i as usize], NUCLEOTIDE_INDEX[j as usize])]
    }
    fn param_count() -> usize {
        GTR_PARAM_N
    }
    fn params(&self) -> &[f64] {
        &self.params
    }
    fn set_param(&mut self, param: usize, value: f64) {
        self.params[param] = value;
        gtr_q(&mut self.q, &self.freqs, &self.params)
    }
    fn param_range(&self, _: usize) -> ParamRange {
        PARAM_RANGE_POSITIVE
    }
    fn freqs(&self) -> &FreqVector {
        &self.freqs
    }
    fn set_freqs(&mut self, freqs: FreqVector) -> Result<()> {
        validate_dna_frequencies(&freqs)?;
        self.freqs = freqs;
        gtr_q(&mut self.q, &self.freqs, &self.params);
        Ok(())
    }
    fn n(&self) -> usize {
        DNA_N
    }
    fn alphabet() -> &'static Alphabet {
        Alphabet::dna()
    }
}

fn gtr_q(q: &mut SubstMatrix, pi: &FreqVector, params: &[f64]) {
    // A <-> G transition rate is fixed to 1.0 to simplify parameter estimation
    // Left in the code for clarity, even though it is fixed to 1.0.
    let ft = pi[0];
    let fc = pi[1];
    let fa = pi[2];
    let fg = pi[3];
    let rtc = params[0];
    let rta = params[1];
    let rtg = params[2];
    let rca = params[3];
    let rcg = params[4];
    let rag = 1.0;

    let scaler = 1.0
        / ((rtc * fc + rta * fa + rtg * fg) * ft
            + (rtc * ft + rca * fa + rcg * fg) * fc
            + (rta * ft + rca * fc + rag * fg) * fa
            + (rtg * ft + rcg * fc + rag * fa) * fg);

    q[(0, 0)] = -(rtc * fc + rta * fa + rtg * fg);
    q[(0, 1)] = rtc * fc;
    q[(0, 2)] = rta * fa;
    q[(0, 3)] = rtg * fg;

    q[(1, 0)] = rtc * ft;
    q[(1, 1)] = -(rtc * ft + rca * fa + rcg * fg);
    q[(1, 2)] = rca * fa;
    q[(1, 3)] = rcg * fg;

    q[(2, 0)] = rta * ft;
    q[(2, 1)] = rca * fc;
    q[(2, 2)] = -(rta * ft + rca * fc + rag * fg);
    q[(2, 3)] = rag * fg;

    q[(3, 0)] = rtg * ft;
    q[(3, 1)] = rcg * fc;
    q[(3, 2)] = rag * fa;
    q[(3, 3)] = -(rtg * ft + rcg * fc + rag * fa);

    q.scale_mut(scaler);
}

impl Display for GTR {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "GTR with [rtc = {:.5}, rta = {:.5}, rtg = {:.5}, rca = {:.5}, rcg = {:.5}, rag = 1.0, freqs = {}]",
            self.params[0], self.params[1], self.params[2],self.params[3],self.params[4], self.freqs
        )
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use assert_matches::assert_matches;
    use rstest::rstest;

    use crate::error::SubstitutionModelError;
    use crate::Error;

    use super::*;

    #[rstest]
    #[case::negative_sum_1(&[-1.0, 1.5, -0.3, 0.8])]
    #[case::all_negative(&[-0.1, -0.2, -0.3, -0.4])]
    #[case::one_negative_sum_1(&[-0.1, 0.5, 0.6, 0.0])]
    fn negative_frequencies(#[case] freqs: &[f64]) {
        assert_matches!(
            validate_dna_frequencies(&frequencies!(freqs)),
            Err(Error::SubstitutionModel(
                SubstitutionModelError::NegativeFrequency { .. }
            ))
        );
    }

    #[rstest]
    #[case::too_few_but_sum_1(&[0.5, 0.4, 0.1])]
    #[case::too_few(&[0.5, 0.4, 0.0])]
    #[case::empty(&[])]
    #[case::one_too_many_but_sum_1(&[0.4, 0.3, 0.1, 0.1, 0.1])]
    #[case::too_many(&[0.1, 0.1, 0.1, 0.1, 0.1, 0.1])]
    #[case::valid_protein(&[1.0 / 20.0; 20])]
    fn wrong_number_of_frequencies(#[case] freqs: &[f64]) {
        match validate_dna_frequencies(&frequencies!(freqs)) {
            Err(Error::SubstitutionModel(SubstitutionModelError::FrequencyCount {
                name,
                actual,
                expected,
            })) => {
                assert_eq!(name, "DNA");
                assert_eq!(actual, freqs.len());
                assert_eq!(expected, DNA_N);
            }
            _ => panic!(
                "Expected FrequencyCount error for DNA with actual = {} and expected = {}",
                freqs.len(),
                DNA_N
            ),
        }
    }

    #[rstest]
    #[case::sum_below_1(&[0.1, 0.2, 0.3, 0.1])]
    #[case::sum_above_1(&[0.4, 0.3, 0.2, 0.2])]
    #[case::sum_above_1_large(&[0.4, 1.3, 0.2, 0.2])]
    fn frequencies_dont_sum_to_1(#[case] freqs: &[f64]) {
        assert_matches!(
            validate_dna_frequencies(&frequencies!(freqs)),
            Err(Error::SubstitutionModel(
                SubstitutionModelError::FrequencySum { .. }
            ))
        );
    }

    #[test]
    fn degenerate_frequencies() {
        for i in 0..DNA_N {
            let mut degenerate_freqs = vec![0.0; DNA_N];
            degenerate_freqs[i] = 1.0;

            match validate_dna_frequencies(&frequencies!(&degenerate_freqs)) {
                Err(Error::SubstitutionModel(SubstitutionModelError::DegenerateFrequencies {
                    name,
                    index,
                })) => {
                    assert_eq!(name, "DNA");
                    assert_eq!(i, index);
                }
                _ => panic!("Expected DegenerateFrequencies error for index {}", i),
            }
        }
    }

    #[test]
    fn jc69_default_params() {
        let model = JC69::default();
        assert_eq!(model.freqs().as_slice(), &EQUAL_FREQS);
        assert_eq!(model.params().len(), JC69::param_count());
    }

    #[test]
    fn k80_default_params() {
        let model = K80::default();
        assert_eq!(model.freqs().as_slice(), &EQUAL_FREQS);
        assert_eq!(model.params().len(), K80::param_count());
        assert_eq!(model.params(), &[2.0]);
    }

    #[test]
    fn hky_default_params() {
        let model = HKY::default();
        assert_eq!(model.freqs().as_slice(), &EQUAL_FREQS);
        assert_eq!(model.params().len(), HKY::param_count());
        assert_eq!(model.params(), &[2.0]);
    }

    #[test]
    fn tn93_default_params() {
        let model = TN93::default();
        assert_eq!(model.freqs().as_slice(), &EQUAL_FREQS);
        assert_eq!(model.params().len(), TN93::param_count());
        assert_eq!(model.params(), &[1.0, 1.0]);
    }

    #[test]
    fn gtr_default_params() {
        let model = GTR::default();
        assert_eq!(model.freqs().as_slice(), &EQUAL_FREQS);
        assert_eq!(model.params().len(), GTR::param_count());
        assert_eq!(model.params(), &[1.0; 5]);
    }
}
