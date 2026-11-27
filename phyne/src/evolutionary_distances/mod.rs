//! Defines the evolutionary-distance trait and implementations for raw
//! Levenshtein distance and model-corrected DNA or protein distances.
//!
//! # Examples
//!
//! ```rust
//! use phyne::evolutionary_distances::{EvolutionaryDistance, Levenshtein};
//! use phyne::record;
//!
//! let a = record!("a", None, b"AGT");
//! let b = record!("b", None, b"ACTT");
//! assert_eq!(Levenshtein.dist(&a, &b), 2.0);
//! ```

use bio::{alignment::distance::levenshtein, io::fasta::Record};

/// A trait representing evolutionary distance measures between two biological sequences,
/// e.g. DNA or protein sequences.
pub trait EvolutionaryDistance {
    /// Calculate the evolutionary distance between two FASTA sequences as defined by the distance measure that
    /// implements this trait.
    fn dist(&self, a: &Record, b: &Record) -> f64;
}

/// Levenshtein distance measure for DNA sequences with Jukes-Cantor correction
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct LevenshteinDNACorrected;

impl EvolutionaryDistance for LevenshteinDNACorrected {
    /// Calculate the evolutionary distance between two DNA sequences using the Levenshtein distance
    /// with Jukes-Cantor correction.
    ///
    /// The Levenshtein distance is the minimum number of single-character edits (insertions, deletions, or
    /// substitutions) required to transform one sequence into the other (e.g. the raw Levenshtein distance between
    /// `AGT` and `ACTT` is 2).
    ///
    /// This implementation applies a Jukes-Cantor model-based correction to the raw Levenshtein distance, based on
    /// formula 1.6 from "Computational Molecular Evolution" by Ziheng Yang (2006):
    ///     d̂ = -3/4 * ln(1 - (4/3) * p̂),
    /// where p̂ is the observed proportion of differing sites (here, Levenshtein distance / max sequence length).
    /// To avoid infinite distances when the sequences are unrelated (i.e., when p̂ approaches 3/4), the proportion
    /// of differing sites is capped at p̂ <= 3/4 - ε.
    ///
    /// # Example:
    /// ```
    /// use phyne::evolutionary_distances::{EvolutionaryDistance, LevenshteinDNACorrected};
    /// use phyne::record;
    ///
    /// let a = record!("a", None, b"AGT");
    /// let b = record!("b", None, b"ACTT");
    /// assert_eq!(LevenshteinDNACorrected.dist(&a, &b), 0.8239592165010822);
    /// ```
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        let seq_i = a.seq();
        let seq_j = b.seq();
        let dist = levenshtein(seq_i, seq_j) as f64;
        let p = f64::min(
            dist / (seq_i.len().max(seq_j.len()) as f64),
            3.0 / 4.0 - f64::EPSILON,
        );
        -(3.0 / 4.0) * (1.0 - (4.0 / 3.0) * p).ln()
    }
}

/// Levenshtein distance measure for protein sequences with Jukes-Cantor-like correction
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct LevenshteinProteinCorrected;

impl EvolutionaryDistance for LevenshteinProteinCorrected {
    /// Calculate the evolutionary distance between two protein sequences using the Levenshtein distance
    /// with Jukes-Cantor-like correction.
    ///
    /// The Levenshtein distance is the minimum number of single-character edits (insertions, deletions, or
    /// substitutions) required to transform one sequence into the other (e.g. the raw Levenshtein distance between
    /// `ARNDCQ` and `RNPCG` is 3).
    ///
    /// This implementation applies a Jukes-Cantor model-based correction to the raw Levenshtein distance, based on
    /// formula 2.3 from "Computational Molecular Evolution" by Ziheng Yang (2006):
    ///     d̂ = -19/20 * ln(1 - (20/19) * p̂),
    /// where p̂ is the observed proportion of differing sites (here, Levenshtein distance / max sequence length).
    /// To avoid infinite distances when the sequences are unrelated (i.e., when p̂ approaches 19/20), the proportion
    /// of differing sites is capped at p̂ <= 19/20 - ε.
    ///
    /// # Example:
    /// ```
    /// use phyne::evolutionary_distances::{EvolutionaryDistance, LevenshteinProteinCorrected};
    /// use phyne::record;
    ///
    /// let a = record!("a", None, b"ARNDCQ");
    /// let b = record!("b", None, b"RNPCG");
    /// assert_eq!(LevenshteinProteinCorrected.dist(&a, &b), 0.70985368173871);
    /// ```
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        let seq_i = a.seq();
        let seq_j = b.seq();
        let dist = levenshtein(seq_i, seq_j) as f64;
        let p = f64::min(
            dist / (seq_i.len().max(seq_j.len()) as f64),
            19.0 / 20.0 - f64::EPSILON,
        );
        -(19.0 / 20.0) * (1.0 - (20.0 / 19.0) * p).ln()
    }
}

/// Levenshtein distance measure for any sequences (raw edit metric)
#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
pub struct Levenshtein;

impl EvolutionaryDistance for Levenshtein {
    /// Calculate the evolutionary distance between two sequences using the raw Levenshtein distance.
    ///
    /// The Levenshtein distance is the minimum number of single-character edits (insertions, deletions, or
    /// substitutions) required to transform one sequence into the other (e.g. the raw Levenshtein distance between
    /// `ARNDCQ` and `RNPCG` is 3).
    ///
    /// # Example:
    /// ```
    /// use phyne::evolutionary_distances::{EvolutionaryDistance, Levenshtein};
    /// use phyne::record;
    ///
    /// let a = record!("a", None, b"ARNDCQ");
    /// let b = record!("b", None, b"RNPCG");
    /// assert_eq!(Levenshtein.dist(&a, &b), 3.0);
    /// ```
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        let seq_i = a.seq();
        let seq_j = b.seq();
        levenshtein(seq_i, seq_j) as f64
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use super::*;

    use crate::record_wo_desc as record;

    #[test]
    fn levenshtein_edit_dist() {
        let dna_corr = Levenshtein {};

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"AAAAAAAA");
        assert_eq!(dna_corr.dist(&s1, &s2), 0.0);
        assert_eq!(dna_corr.dist(&s1, &s1), 0.0);

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"TTTTTTTT");
        assert_eq!(dna_corr.dist(&s1, &s2), s1.seq().len() as f64);

        let s1 = record!("s1", b"AAAAAAAAAAAAAAAA");
        let s2 = record!("s2", b"AAAAA");
        assert_eq!(
            dna_corr.dist(&s1, &s2),
            (s1.seq().len() - s2.seq().len()) as f64
        );

        let s1 = record!("s1", b"AAAAATAACAAAGTAAA");
        let s2 = record!("s2", b"AAAAATCAGT");
        assert_eq!(
            dna_corr.dist(&s1, &s2),
            (s1.seq().len() - s2.seq().len()) as f64
        );

        let s1 = record!("s1", b"AAAAAAAAAAAAAAAA");
        let s2 = record!("s2", b"TTTTT");
        assert_eq!(dna_corr.dist(&s1, &s2), s1.seq().len() as f64);
    }

    #[test]
    fn levenshtein_dna_corrected_corners() {
        let dna_corr = LevenshteinDNACorrected {};

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"AAAAAAAA");
        assert_eq!(dna_corr.dist(&s1, &s2), 0.0);
        assert_eq!(dna_corr.dist(&s1, &s1), 0.0);

        let max_proportion = 3.0 / 4.0;
        let max_distance =
            -max_proportion * (1.0 - 1.0 / max_proportion * (max_proportion - f64::EPSILON)).ln();
        // Max distance possible between sequences (all chars are different), computed with Python
        assert_eq!(max_distance, 26.728641210756745);

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"TTTTTTTT");
        assert_eq!(dna_corr.dist(&s1, &s2), max_distance);

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"");
        assert_eq!(dna_corr.dist(&s1, &s2), max_distance);
    }

    #[test]
    fn levenshtein_dna_corrected() {
        let dna_corr = LevenshteinDNACorrected {};

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"AAAAAAAT");
        // Levenshtein distance is 1, proportion is 1/8=0.125, distance computed with Python
        assert_eq!(dna_corr.dist(&s1, &s2), 0.13674116759546595);

        let s1 = record!("s1", b"AAAAATAACAAAGTAAA");
        let s2 = record!("s2", b"AAAAATCAGT");
        // Levenshtein distance is 7, proportion is 7/17=0.4117647058823529, distance computed with Python
        assert_eq!(dna_corr.dist(&s1, &s2), 0.5972485625963819);
    }

    #[test]
    fn levenshtein_dna_corrected_sanity() {
        let dna_corr = LevenshteinDNACorrected {};

        let s1 = record!("seq1", b"ACGTACGTXXXXXX");
        let s2 = record!("seq2", b"AAGTACGTXXXXXX");
        let s3 = record!("seq3", b"AAGTTCGTXXXXXX");
        let s4 = record!("seq4", b"TTTTTTTTXXXXXX");

        assert!(dna_corr.dist(&s1, &s2) < dna_corr.dist(&s1, &s3));
        assert!(dna_corr.dist(&s1, &s2) < dna_corr.dist(&s1, &s4));
        assert!(dna_corr.dist(&s1, &s3) < dna_corr.dist(&s1, &s4));
        assert!(dna_corr.dist(&s2, &s3) < dna_corr.dist(&s2, &s4));
        assert_eq!(dna_corr.dist(&s1, &s4), dna_corr.dist(&s2, &s4));
        assert!(dna_corr.dist(&s3, &s4) < dna_corr.dist(&s2, &s4));

        let s5 = record!("seq5", b"GGGGGGGGGGGGGG");
        assert_eq!(dna_corr.dist(&s1, &s5), dna_corr.dist(&s2, &s5));
        assert_eq!(dna_corr.dist(&s1, &s5), dna_corr.dist(&s3, &s5));
        assert_eq!(dna_corr.dist(&s1, &s5), dna_corr.dist(&s4, &s5));
    }

    #[test]
    fn levenshtein_protein_corrected_corners() {
        let prot_corr = LevenshteinProteinCorrected {};

        let s1 = record!("s1", b"PPPPPPPP");
        let s2 = record!("s2", b"PPPPPPPP");
        assert_eq!(prot_corr.dist(&s1, &s2), 0.0);
        assert_eq!(prot_corr.dist(&s1, &s1), 0.0);

        let max_proportion = 19.0 / 20.0;
        let max_distance =
            -max_proportion * (1.0 - 1.0 / max_proportion * (max_proportion - f64::EPSILON)).ln();
        // Max distance possible between sequences (all chars are different), computed with Python
        assert_eq!(max_distance, 33.85627886695854);

        let s1 = record!("s1", b"PPPPPPPP");
        let s2 = record!("s2", b"NNNNNNNN");
        assert_eq!(prot_corr.dist(&s1, &s2), max_distance);

        let s1 = record!("s1", b"PPPPPPPP");
        let s2 = record!("s2", b"");
        assert_eq!(prot_corr.dist(&s1, &s2), max_distance);
    }

    #[test]
    fn levenshtein_protein_corrected() {
        let prot_corr = LevenshteinProteinCorrected {};

        let s1 = record!("s1", b"AAAAAAAA");
        let s2 = record!("s2", b"AAAAAAAT");
        // Levenshtein distance is 1, proportion is 1/8=0.125, distance computed with Python
        assert_eq!(prot_corr.dist(&s1, &s2), 0.1340246683469102);

        let s1 = record!("s1", b"AAAAATAACAAAGTAAA");
        let s2 = record!("s2", b"AAAAATCAGT");
        // Levenshtein distance is 7, proportion is 7/17=0.4117647058823529, distance computed with Python
        assert_eq!(prot_corr.dist(&s1, &s2), 0.5397578618621736);
    }

    #[test]
    fn levenshtein_protein_corrected_sanity() {
        let prot_corr = LevenshteinProteinCorrected {};

        let s1 = record!("seq1", b"ARNDCQEGHILKMFPSTWYV");
        let s2 = record!("seq2", b"ARNDCQEGHILKMFPSTWVV");
        let s3 = record!("seq3", b"ARNDCQEGHILKMFPSRRVV");
        let s4 = record!("seq4", b"RRRRRRRRRRRRRRRRRRRR");

        assert!(prot_corr.dist(&s1, &s2) < prot_corr.dist(&s1, &s3));
        assert!(prot_corr.dist(&s1, &s2) < prot_corr.dist(&s1, &s4));
        assert!(prot_corr.dist(&s1, &s3) < prot_corr.dist(&s1, &s4));
        assert!(prot_corr.dist(&s2, &s3) < prot_corr.dist(&s2, &s4));
        assert!(prot_corr.dist(&s3, &s4) < prot_corr.dist(&s2, &s4));
        assert_eq!(prot_corr.dist(&s1, &s4), prot_corr.dist(&s2, &s4));
        assert!(prot_corr.dist(&s1, &s4) > prot_corr.dist(&s3, &s4));
    }
}
