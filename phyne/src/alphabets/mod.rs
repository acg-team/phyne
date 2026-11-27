//! Module for handling biological sequence alphabets in phylogenetic analyses.
//!
//! Contains definitions for a generic alphabet, as well as specific alphabets for DNA
//! and protein sequences. The specific alphabets include cached conditional probabilities and
//! parsimony sets to speed up likelihood and parsimony score calculations.
//!
//! The DNA and protein alphabets can be accessed as static instances using `Alphabet::dna()`
//! and `Alphabet::protein()`.
//!
//! # Examples
//!
//! ```rust
//! use phyne::alphabets::Alphabet;
//!
//! let dna = Alphabet::dna();
//! assert_eq!(dna.name(), "DNA");
//! assert!(dna.is_word(b"aCgT"));
//! ```

use std::fmt::Display;

use hashbrown::HashSet;
use lazy_static::lazy_static;
use nalgebra::DVector;

use crate::evolutionary_models::FreqVector;
use crate::frequencies;

#[doc(hidden)]
pub mod parsimony_set;
#[doc(inline)]
pub use parsimony_set::*;

/// Type alias for conditional probability vectors.
/// E.g. for the DNA alphabet, the R character (a purine) would have a conditional probability vector of
/// `[0.0, 0.0, 1.0, 1.0]`, indicating the probabilities for each nucleotide (T, C, A, G) respectively.
pub type ConditionalProbs = DVector<f64>;

/// All single-letter symbols for valid non-ambiguous amino acids used in the alphabet.
pub static AMINOACIDS: &[u8] = b"ARNDCQEGHILKMFPSTWYV";

/// All single-letter symbols for ambiguous amino acids used in the alphabet.
pub static AMB_AMINOACIDS: &[u8] = b"BJZX";

/// All single-letter symbols for valid non-ambiguous nucleotides used in the alphabet.
/// This is legacy sorting, i.e., conforming with tools like RaXML and PhyML, and is used also to maintain the order
/// of rows/columns in substitution matrices.
pub static NUCLEOTIDES: &[u8] = b"TCAG";

/// All single-letter symbols for ambiguous nucleotides used in the alphabet.
pub static AMB_NUCLEOTIDES: &[u8] = b"RYSWKMBDHVNZX";

/// The character representing an ambiguous symbol in both the amino acid and nucleotide alphabets.
pub static AMB_CHAR: u8 = b'X';

/// The character representing a gap in an [`MSA`](`crate::alignment::MSA`) or [`MASA`](`crate::alignment::MASA`).
pub static GAP: u8 = b'-';

/// All possible characters that could represent a gap in an MSA, included for compatibility with other tools.
pub static POSSIBLE_GAPS: &[u8] = b"_*-";

/// Represents a biological sequence alphabet, including valid symbols, ambiguous symbols, and
/// precomputed data for efficient sequence analysis (conditional probabilities and parsimony set
/// definitions for each valid symbol).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alphabet {
    /// Human-readable name of the alphabet.
    name: &'static str,
    /// Unambiguous symbols in the alphabet in the same order as used in substitution matrices (e.g. for DNA, the order
    /// is [`TCAG`](`NUCLEOTIDES`), and the rate of change from A to C will be stored in the instantaneous rate matrix
    /// at `Q[2][1]`).
    symbols: &'static [u8],
    /// Symbols that represent more than one unambiguous symbol.
    ambiguous: &'static [u8],
    /// Mapping from ASCII character codes to symbol indices for efficient lookups.
    index: &'static [usize; 255],
    /// Symbols accepted by the alphabet, including ambiguous symbols and gaps.
    valid_symbols: &'static HashSet<u8>,
    /// Conditional probability vectors indexed by character code.
    conditional_probs: &'static [FreqVector],
    /// Parsimony sets indexed by character code.
    parsimony_sets: &'static [ParsimonySet],
}

impl Display for Alphabet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let capitalised_name = {
            let mut c = self.name.chars();
            match c.next() {
                None => String::new(),
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
            }
        };
        writeln!(
            f,
            "{} sequence alphabet of length {}",
            capitalised_name,
            self.len()
        )?;
        writeln!(
            f,
            "Valid symbols: {}",
            String::from_utf8_lossy(self.symbols)
        )?;
        writeln!(f, "Ambiguous symbols:")?;
        for &char in self.ambiguous {
            writeln!(
                f,
                "\t{}: {} ",
                char as char, self.parsimony_sets[char as usize]
            )?;
        }
        writeln!(
            f,
            "Possible gap representations: {}",
            String::from_utf8_lossy(POSSIBLE_GAPS)
        )
    }
}

impl Alphabet {
    /// Returns the name of the alphabet.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert_eq!(Alphabet::dna().name(), "DNA");
    /// assert_eq!(Alphabet::protein().name(), "protein");
    /// ```
    pub fn name(&self) -> &str {
        self.name
    }

    /// Returns the DNA alphabet as a static reference (`&'static Alphabet`)
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// let alphabet = Alphabet::dna();
    /// assert_eq!(alphabet.len(), 4);
    /// assert_eq!(alphabet.name(), "DNA");
    /// ```
    pub fn dna() -> &'static Self {
        &DNA_ALPHABET
    }

    /// Returns the protein alphabet as a static reference (`&'static Alphabet`).
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// let alphabet = Alphabet::protein();
    /// assert_eq!(alphabet.len(), 20);
    /// assert_eq!(alphabet.name(), "protein");
    /// ```
    pub fn protein() -> &'static Self {
        &PROTEIN_ALPHABET
    }

    /// Checks if a word is valid in this alphabet, case-insensitive.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert!(Alphabet::dna().is_word(b"ACGT"));
    /// assert!(Alphabet::dna().is_word(b"aCgTx"));
    /// assert!(Alphabet::protein().is_word(b"ACGTFTH"));
    /// assert!(!Alphabet::dna().is_word(b"ACGTFTH"));
    /// ```
    pub fn is_word(&self, word: &[u8]) -> bool {
        word.to_ascii_uppercase()
            .iter()
            .all(|c| self.valid_symbols.contains(c))
    }

    /// Returns the valid and unambiguous symbols of the alphabet.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert_eq!(Alphabet::dna().symbols(), b"TCAG");
    /// assert_eq!(Alphabet::protein().symbols(), b"ARNDCQEGHILKMFPSTWYV");
    /// ```
    pub fn symbols(&self) -> &[u8] {
        self.symbols
    }

    /// Returns the ambiguous symbols of the alphabet.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert_eq!(Alphabet::dna().ambiguous(), b"RYSWKMBDHVNZX");
    /// assert_eq!(Alphabet::protein().ambiguous(), b"BJZX");
    /// ```
    pub fn ambiguous(&self) -> &[u8] {
        self.ambiguous
    }

    /// Returns the number of symbols in the alphabet, not including ambiguous characters or gaps.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert_eq!(Alphabet::dna().len(), 4);
    /// assert_eq!(Alphabet::protein().len(), 20);
    /// ```
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Returns the index mapping for the alphabet.
    /// The index maps ASCII character codes to their respective indices in the alphabet, case-insensitive.
    /// Used to maintain consistent ordering in frequency vectors and substitution matrices.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::Alphabet;
    /// assert_eq!(Alphabet::dna().index(&b'T'), 0);
    /// assert_eq!(Alphabet::dna().index(&b'c'), 1);
    /// assert_eq!(Alphabet::dna().index(&b'A'), 2);
    /// assert_eq!(Alphabet::dna().index(&b'g'), 3);
    /// assert_eq!(Alphabet::protein().index(&b'R'), 1);
    /// assert_eq!(Alphabet::protein().index(&b'w'), 17);
    /// ```
    pub fn index(&self, char: &u8) -> usize {
        self.index[*char as usize]
    }

    /// Returns the conditional probability vector for a given character in the alphabet.
    /// The vector represents the conditional probabilities of observing each symbol in the alphabet given the specified character.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::{Alphabet, ConditionalProbs};
    /// let t_probs = Alphabet::dna().char_encoding(b'T');
    /// assert_eq!(t_probs, &ConditionalProbs::from_vec(vec![1.0, 0.0, 0.0, 0.0]));
    /// let amb_probs = Alphabet::dna().char_encoding(b'X');
    /// assert_eq!(amb_probs, &ConditionalProbs::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
    /// ```
    pub fn char_encoding(&self, char: u8) -> &ConditionalProbs {
        &self.conditional_probs[char.to_ascii_uppercase() as usize]
    }

    /// Returns the conditional probability vector for the gap character in the alphabet, encoded as missing data.
    ///
    /// # Example:
    /// ```
    /// # use phyne::alphabets::{Alphabet, ConditionalProbs};
    /// let gap_probs = Alphabet::dna().missing_char_encoding();
    /// assert_eq!(gap_probs, &ConditionalProbs::from_vec(vec![1.0, 1.0, 1.0, 1.0]));
    /// let amb_probs = Alphabet::dna().char_encoding(b'X');
    /// assert_eq!(gap_probs, amb_probs);
    /// ```
    pub fn missing_char_encoding(&self) -> &ConditionalProbs {
        &self.conditional_probs[AMB_CHAR as usize]
    }

    /// Returns the parsimony set for a given character in the alphabet.
    /// The parsimony set represents the set of unambiguous symbols that the character can represent.
    /// For example, in the DNA alphabet, the character 'R' represents the set {'A', 'G'}.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::{Alphabet, ParsimonySet};
    /// let r_set = Alphabet::dna().parsimony_set(&b'R');
    /// assert_eq!(format!("{r_set}"), "[AG]");
    /// let z_set = Alphabet::protein().parsimony_set(&b'Z');
    /// assert_eq!(format!("{z_set}"), "[EQ]");
    /// ```
    pub fn parsimony_set(&self, char: &u8) -> &ParsimonySet {
        &self.parsimony_sets[*char as usize]
    }

    /// Returns the parsimony set representing a gap character in the alphabet.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::{Alphabet, ParsimonySet};
    /// let gap_set = Alphabet::dna().gap_set();
    /// assert_eq!(format!("{gap_set}"), "[-]");
    /// ```
    pub fn gap_set(&self) -> &ParsimonySet {
        &GAP_SET
    }

    /// Returns the parsimony set representing the characters in the given slice within the respective alphabet.
    ///
    /// # Example:
    /// ```
    /// use phyne::alphabets::{Alphabet, ParsimonySet};
    /// let set_all = Alphabet::dna().parsimony_set_from_slice(b"CGAT");
    /// let set_amb = Alphabet::dna().parsimony_set_from_slice(b"X");
    /// assert_eq!(set_all, set_amb);
    /// let set_incomplete = Alphabet::dna().parsimony_set_from_slice(b"CG");
    /// assert_ne!(set_all, set_incomplete);
    /// ```
    pub fn parsimony_set_from_slice(&self, slice: &[u8]) -> ParsimonySet {
        slice
            .iter()
            .map(|&c| self.parsimony_set(&c))
            .fold(ParsimonySet::empty(), |acc, x| &acc | x)
    }
}

lazy_static! {
    /// Static standard DNA alphabet definition
    pub static ref DNA_ALPHABET: Alphabet = Alphabet {
        name: "DNA",
        symbols: NUCLEOTIDES,
        ambiguous: AMB_NUCLEOTIDES,
        index: &NUCLEOTIDE_INDEX,
        valid_symbols: &VALID_NUCLEOTIDES,
        conditional_probs: &NUCL_COND_PROBS,
        parsimony_sets: &NUCL_PARSIMONY_SETS,
    };

    /// Static nucleotide index mapping for quick lookup, mapping each nucleotide character to its
    /// index in the DNA alphabet.
    pub(crate) static ref NUCLEOTIDE_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, char) in NUCLEOTIDES.iter().enumerate() {
            index[*char as usize] = i;
            index[(*char).to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 4;
        index
    };

    /// Static set of valid nucleotide characters, including ambiguous nucleotides and the gap character.
    pub(crate) static ref VALID_NUCLEOTIDES: HashSet<u8> = {
        NUCLEOTIDES
            .iter()
            .chain(AMB_NUCLEOTIDES.iter().chain([GAP].iter()))
            .cloned()
            .collect()
    };

    /// Static conditional probability vectors for each nucleotide character, used in e.g. Felsenstein's
    /// pruning algorithm.
    pub(crate) static ref NUCL_COND_PROBS: Vec<FreqVector> = {
        let mut map = vec![frequencies!(&[0.0; 4]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &nucl_cond_probs(char));
        }
        map
    };

    /// Static parsimony sets for each nucleotide character, used in parsimony-based phylogenetic inference.
    pub(crate) static ref NUCL_PARSIMONY_SETS: Vec<ParsimonySet> = {
        let mut map: Vec<ParsimonySet> = vec![ParsimonySet::empty(); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            *elem = nucl_parsimony_set(&char);
        }
        map
    };

    /// Static parsimony set for the gap character for e.g. Dollo parsimony.
    pub(crate) static ref GAP_SET: ParsimonySet = ParsimonySet::gap();
}

/// Function used to precompute static conditional probability vectors for each nucleotide character.
fn nucl_cond_probs(char: u8) -> FreqVector {
    let char = char.to_ascii_uppercase();
    match char {
        b'T' => frequencies!(&[1.0, 0.0, 0.0, 0.0]),
        b'C' => frequencies!(&[0.0, 1.0, 0.0, 0.0]),
        b'A' => frequencies!(&[0.0, 0.0, 1.0, 0.0]),
        b'G' => frequencies!(&[0.0, 0.0, 0.0, 1.0]),
        b'M' => frequencies!(&[0.0, 1.0, 1.0, 0.0]),
        b'R' => frequencies!(&[0.0, 0.0, 1.0, 1.0]),
        b'W' => frequencies!(&[1.0, 0.0, 1.0, 0.0]),
        b'S' => frequencies!(&[0.0, 1.0, 0.0, 1.0]),
        b'Y' => frequencies!(&[1.0, 1.0, 0.0, 0.0]),
        b'K' => frequencies!(&[1.0, 0.0, 0.0, 1.0]),
        b'V' => frequencies!(&[0.0, 1.0, 1.0, 1.0]),
        b'D' => frequencies!(&[1.0, 0.0, 1.0, 1.0]),
        b'B' => frequencies!(&[1.0, 1.0, 0.0, 1.0]),
        b'H' => frequencies!(&[1.0, 1.0, 1.0, 0.0]),
        _ => frequencies!(&[1.0; 4]),
    }
}

/// Function used to precompute static parsimony sets for each nucleotide character.
fn nucl_parsimony_set(char: &u8) -> ParsimonySet {
    let char = char.to_ascii_uppercase();
    if NUCLEOTIDES.contains(&char) {
        return ParsimonySet::from_slice(&[char]);
    }
    match char {
        b'-' => ParsimonySet::from_slice(&[GAP]),
        b'M' => ParsimonySet::from_slice(b"CA"),
        b'R' => ParsimonySet::from_slice(b"AG"),
        b'W' => ParsimonySet::from_slice(b"TA"),
        b'S' => ParsimonySet::from_slice(b"CG"),
        b'Y' => ParsimonySet::from_slice(b"TC"),
        b'K' => ParsimonySet::from_slice(b"TG"),
        b'V' => ParsimonySet::from_slice(b"CAG"),
        b'D' => ParsimonySet::from_slice(b"TAG"),
        b'B' => ParsimonySet::from_slice(b"TCG"),
        b'H' => ParsimonySet::from_slice(b"TCA"),
        _ => ParsimonySet::from_slice(NUCLEOTIDES),
    }
}

lazy_static! {
    /// Static standard protein alphabet definition
    pub static ref PROTEIN_ALPHABET: Alphabet = Alphabet {
        name: "protein",
        symbols: AMINOACIDS,
        ambiguous: AMB_AMINOACIDS,
        index: &AMINOACID_INDEX,
        valid_symbols: &VALID_AMINOACIDS,
        conditional_probs: &AA_COND_PROBS,
        parsimony_sets: &AA_PARSIMONY_SETS,
    };

    /// Static amino acid index mapping for quick lookup, mapping each character to its
    /// index in the protein alphabet.
    pub(crate) static ref AMINOACID_INDEX: [usize; 255] = {
        let mut index = [0; 255];
        for (i, &char) in AMINOACIDS.iter().enumerate() {
            index[char as usize] = i;
            index[char.to_ascii_lowercase() as usize] = i;
        }
        index[GAP as usize] = 20;
        index
    };

    /// Static set of valid amino acid characters, including ambiguous amino acids and the gap character.
    pub(crate) static ref VALID_AMINOACIDS: HashSet<u8> = {
        AMINOACIDS
            .iter()
            .chain(AMB_AMINOACIDS.iter().chain([GAP].iter()))
            .cloned()
            .collect()
    };

    /// Static conditional probability vectors for each amino acid character, used in e.g. Felsenstein's
    /// pruning algorithm.
    pub(crate) static ref AA_COND_PROBS: Vec<FreqVector> = {
        let mut map: Vec<FreqVector> = vec![frequencies!(&[0.0; 20]); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            elem.set_column(0, &aa_cond_probs(char));
        }
        map
    };

    /// Static parsimony sets for each amino acid character, used in parsimony-based phylogenetic inference.
    pub(crate) static ref AA_PARSIMONY_SETS: Vec<ParsimonySet> = {
        let mut map: Vec<ParsimonySet> = vec![ParsimonySet::empty(); 255];
        for (i, elem) in map.iter_mut().enumerate() {
            let char = i as u8;
            *elem = aa_parsimony_set(&char);
        }
        map
    };
}

/// Function used to precompute static conditional probability vectors for each amino acid character.
fn aa_cond_probs(char: u8) -> FreqVector {
    let char = char.to_ascii_uppercase();
    let index = &AMINOACID_INDEX;
    if AMINOACIDS.contains(&char) {
        let mut set = frequencies!(&[0.0; 20]);
        set.fill_row(index[char as usize], 1.0);
        return set;
    }
    match char {
        b'B' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'D' as usize], 1.0);
            set.fill_row(index[b'N' as usize], 1.0);
            set
        }
        b'Z' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'E' as usize], 1.0);
            set.fill_row(index[b'Q' as usize], 1.0);
            set
        }
        b'J' => {
            let mut set = frequencies!(&[0.0; 20]);
            set.fill_row(index[b'I' as usize], 1.0);
            set.fill_row(index[b'L' as usize], 1.0);
            set
        }
        _ => {
            frequencies!(&[1.0; 20])
        }
    }
}

/// Function used to precompute static parsimony sets for each amino acid character.
fn aa_parsimony_set(char: &u8) -> ParsimonySet {
    let char = char.to_ascii_uppercase();
    if AMINOACIDS.contains(&char) {
        return ParsimonySet::from_slice(&[char]);
    }
    match char {
        b'-' => ParsimonySet::from_slice(&[GAP]),
        b'B' => ParsimonySet::from_slice(b"DN"),
        b'Z' => ParsimonySet::from_slice(b"EQ"),
        b'J' => ParsimonySet::from_slice(b"IL"),
        _ => ParsimonySet::from_slice(AMINOACIDS),
    }
}

#[cfg(test)]
lazy_static! {
    /// Static empty set of symbols, used for the unknown alphabet.
    static ref EMPTY_SYMBOLS: HashSet<u8> = HashSet::new();

    /// Static definition of an unknown alphabet, can be used as a fallback when the alphabet of a
    /// sequence is not known (currently only used for testing).
    pub static ref UNKNOWN_ALPHABET: Alphabet = Alphabet {
        name: "unknown",
        symbols: &[],
        ambiguous: &[],
        index: &[0; 255],
        valid_symbols: &EMPTY_SYMBOLS,
        conditional_probs: &[],
        parsimony_sets: &[]
    };
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
