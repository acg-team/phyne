use bio::{alignment::distance::levenshtein, io::fasta::Record};
use nalgebra::max;

pub trait EvolutionaryDistance {
    fn dist(&self, a: &Record, b: &Record) -> f64;
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
/// Levenshtein distance with Jukes-Cantor correction, meaningful for DNA sequences
pub struct LevenshteinDNACorrected;

impl EvolutionaryDistance for LevenshteinDNACorrected {
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        // Distance formula corrected using the Jukes-Cantor model
        // Formula 1.6 from "Computational Molecular Evolution" by Ziheng Yang (2006)
        //     d = -3/4 * ln(1 - (4/3) * p)
        // where p is the proportion of differing sites (here, Levenshtein distance / max length).
        // To avoid infinite distance when all characters are different, the maximum
        // proportion of different characters is capped to 3/4=0.75.
        let seq_i = a.seq();
        let seq_j = b.seq();
        let dist = levenshtein(seq_i, seq_j) as f64;
        let p = f64::min(
            dist / (max(seq_i.len(), seq_j.len()) as f64),
            3.0 / 4.0 - f64::EPSILON,
        );
        -(3.0 / 4.0) * (1.0 - (4.0 / 3.0) * p).ln()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
/// Levenshtein distance with Jukes-Cantor-like correction, meaningful for protein sequences
pub struct LevenshteinProteinCorrected;

impl EvolutionaryDistance for LevenshteinProteinCorrected {
    fn dist(&self, a: &Record, b: &Record) -> f64 {
        // Corrected protein distance formula equivalent to Jukes-Cantor for proteins.
        // Formula 2.3 from "Computational Molecular Evolution" by Ziheng Yang (2006)
        //     d = -19/20 * ln(1 - (20/19) * p)
        // where p is the proportion of differing sites (here, Levenshtein distance / max length).
        // To avoid infinite distance when all characters are different, the maximum
        // proportion of different characters is capped to 19/20=0.95.

        let seq_i = a.seq();
        let seq_j = b.seq();
        let dist = levenshtein(seq_i, seq_j) as f64;
        let p = f64::min(
            dist / (max(seq_i.len(), seq_j.len()) as f64),
            19.0 / 20.0 - f64::EPSILON,
        );
        -(19.0 / 20.0) * (1.0 - (20.0 / 19.0) * p).ln()
    }
}

#[derive(Clone, Copy, Debug, Default, Eq, Hash, PartialEq, PartialOrd, Ord)]
/// Simple Levenshtein distance without any correction, meaningful for both DNA
/// and protein sequences
pub struct Levenshtein;

impl EvolutionaryDistance for Levenshtein {
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
