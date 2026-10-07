use std::fmt::Display;
use std::ops::{Index, IndexMut};
use std::slice;

use bio::io::fasta::Record;
use bitvec::vec::BitVec;
use hashbrown::HashSet;

use crate::alphabets::{Alphabet, GAP};
use crate::{bail, record, Result};

/// Container for a set of sequences, which may or may not be aligned.
///
/// This struct holds a collection of `bio::io::fasta::Record`s and provides methods
/// for managing them, including alphabet detection and validation of sequence ID uniqueness.
/// Tracks whether the sequences are currently aligned (all have the same length).
#[derive(Debug, Clone)]
pub struct Sequences {
    pub(crate) s: Vec<Record>,
    pub(crate) aligned: bool,
    pub(crate) alphabet: &'static Alphabet,
}

impl PartialEq for Sequences {
    fn eq(&self, other: &Self) -> bool {
        if self.s.len() != other.s.len()
            || self.aligned != other.aligned
            || self.alphabet != other.alphabet
        {
            return false;
        }

        let mut self_refs: Vec<&Record> = self.s.iter().collect();
        let mut other_refs: Vec<&Record> = other.s.iter().collect();

        self_refs.sort_by_key(|r| r.id());
        other_refs.sort_by_key(|r| r.id());

        self_refs == other_refs
    }
}

impl Display for Sequences {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for record in &self.s {
            write!(f, "{record}")?;
        }
        Ok(())
    }
}

impl Index<usize> for Sequences {
    type Output = Record;

    fn index(&self, index: usize) -> &Self::Output {
        &self.s[index]
    }
}

impl IndexMut<usize> for Sequences {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.s[index]
    }
}

impl<'a> IntoIterator for &'a Sequences {
    type Item = &'a Record;
    type IntoIter = slice::Iter<'a, Record>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Sequences {
    /// Creates a new `Sequences` object from a vector of `bio::io::fasta::Record`.
    ///
    /// The alphabet is automatically detected from the sequences.
    /// The `Sequences` object is considered aligned if all sequences have the same length.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![
    ///     record!("seq1", None, b"ACGT"),
    ///     record!("seq2", None, b"ACGT"),
    /// ];
    /// let seqs = Sequences::new(records)?;
    /// assert_eq!(seqs.len(), 2);
    /// # Ok(()) }
    /// ```
    pub fn new(s: Vec<Record>) -> Result<Sequences> {
        let alphabet = detect_alphabet(&s);
        Self::with_alphabet(s, alphabet)
    }

    /// Creates a new `Sequences` object from a vector of `bio::io::fasta::Record` and a provided alphabet.
    ///
    /// The `Sequences` object is considered aligned if all sequences have the same length.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::alphabets::Alphabet;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"ACGT")];
    /// let seqs = Sequences::with_alphabet(records, Alphabet::dna())?;
    /// assert_eq!(seqs.alphabet(), Alphabet::dna());
    /// # Ok(()) }
    /// ```
    pub fn with_alphabet(s: Vec<Record>, alphabet: &'static Alphabet) -> Result<Sequences> {
        let potential_msa_len = if s.is_empty() { 0 } else { s[0].seq().len() };
        // Sequences are aligned if all sequences are the same length
        let aligned = s.iter().skip(1).all(|r| r.seq().len() == potential_msa_len);

        let seqs = Sequences {
            s,
            aligned,
            alphabet,
        };
        seqs.ids_are_unique()?;
        Ok(seqs)
    }

    /// Creates a new `Sequences` object from a vector of `bio::io::fasta::Record`.
    ///
    /// The alphabet is automatically detected from the sequences.
    /// The `Sequences` object is considered aligned if all sequences have the same length.
    /// The unchecked version is meant for test purposes only and does not check that
    /// the ids are unique.
    #[cfg(test)]
    pub(crate) fn new_unchecked(s: Vec<Record>) -> Sequences {
        let alphabet = detect_alphabet(&s);
        Self::with_alphabet_unchecked(s, alphabet)
    }

    /// Creates a new `Sequences` object from a vector of `bio::io::fasta::Record` and a provided alphabet.
    ///
    /// The `Sequences` object is considered aligned if all sequences have the same length.
    /// The unchecked version is meant for recreating the object from an existing valid one
    /// and does not check that the ids are unique.
    pub(crate) fn with_alphabet_unchecked(
        s: Vec<Record>,
        alphabet: &'static Alphabet,
    ) -> Sequences {
        let potential_msa_len = if s.is_empty() { 0 } else { s[0].seq().len() };
        // Sequences are aligned if all sequences are the same length
        let aligned = s.iter().skip(1).all(|r| r.seq().len() == potential_msa_len);

        Sequences {
            s,
            aligned,
            alphabet,
        }
    }

    /// Returns an iterator over the sequences.
    fn iter(&self) -> slice::Iter<'_, Record> {
        self.s.iter()
    }

    /// Returns the number of sequences.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let seqs = Sequences::new(vec![])?;
    /// assert_eq!(seqs.len(), 0);
    /// # Ok(()) }
    /// ```
    pub fn len(&self) -> usize {
        self.s.len()
    }

    /// Returns `true` if there are no sequences.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let seqs = Sequences::new(vec![])?;
    /// assert!(seqs.is_empty());
    /// # Ok(()) }
    /// ```
    pub fn is_empty(&self) -> bool {
        self.s.is_empty()
    }

    /// Returns a reference to the record with the given ID.
    ///
    /// # Panics
    ///
    /// Panics if no sequence with the given ID is found.
    /// Use [`Sequences::try_record_by_id`] for a non-panicking version.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"A")];
    /// let seqs = Sequences::new(records)?;
    /// let rec = seqs.record_by_id("seq1");
    /// assert_eq!(rec.id(), "seq1");
    /// # Ok(()) }
    /// ```
    pub fn record_by_id(&self, id: &str) -> &Record {
        self.s
            .iter()
            .find(|r| r.id() == id)
            .unwrap_or_else(|| panic!("Sequence with id {id} not found"))
    }

    /// Replaces the record with the given ID with a new record.
    ///
    /// # Errors
    /// Returns an error if no record with the given ID is found.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"A")];
    /// let mut seqs = Sequences::new(records)?;
    /// let new_record = record!("seq1", None, b"C");
    /// seqs.update_record("seq1", new_record)?;
    /// assert_eq!(seqs.record_by_id("seq1").seq(), b"C");
    /// # Ok(()) }
    /// ```
    pub fn update_record(&mut self, id: &str, new_record: Record) -> Result<()> {
        let idx = self.s.iter().position(|r| r.id() == id);
        match idx {
            Some(i) => {
                self.s[i] = new_record;
                Ok(())
            }
            None => bail!(Sequence, "sequence with id {id} not found"),
        }
    }

    /// Returns a reference to the record with the given ID, or an error if not found.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"A")];
    /// let seqs = Sequences::new(records)?;
    /// assert!(seqs.try_record_by_id("seq1").is_ok());
    /// assert!(seqs.try_record_by_id("seq2").is_err());
    /// # Ok(()) }
    /// ```
    pub fn try_record_by_id(&self, id: &str) -> Result<&Record> {
        let rec = self.s.iter().find(|r| r.id() == id);
        match rec {
            Some(r) => Ok(r),
            None => bail!(Sequence, "sequence with id {id} not found"),
        }
    }

    /// Returns the alphabet of the sequences.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::alphabets::Alphabet;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"A")];
    /// let seqs = Sequences::new(records)?;
    /// assert_eq!(seqs.alphabet(), Alphabet::dna());
    /// # Ok(()) }
    /// ```
    pub fn alphabet(&self) -> &'static Alphabet {
        self.alphabet
    }

    /// Removes all gaps from the sequences and returns a new `Sequences` object.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![record!("seq1", None, b"A-C")];
    /// let seqs = Sequences::new(records)?;
    /// let gapless = seqs.into_gapless();
    /// assert_eq!(gapless.record_by_id("seq1").seq(), b"AC");
    /// # Ok(()) }
    /// ```
    pub fn into_gapless(&self) -> Sequences {
        let seqs = self
            .s
            .iter()
            .map(|rec| {
                let sequence = rec
                    .seq()
                    .iter()
                    .filter(|&c| c != &GAP)
                    .copied()
                    .collect::<Vec<u8>>();
                record!(rec.id(), rec.desc(), &sequence)
            })
            .collect();
        Sequences {
            s: seqs,
            aligned: false,
            alphabet: self.alphabet,
        }
    }

    /// Removes all columns that only contain gaps from the sequences.
    ///
    /// # Panics
    ///
    /// Panics if the sequences are not aligned.
    ///
    /// # Example:
    /// ```
    /// use phylo::alignment::Sequences;
    /// use phylo::record;
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let records = vec![
    ///     record!("seq1", None, b"A-C"),
    ///     record!("seq2", None, b"T-G"),
    /// ];
    /// let mut seqs = Sequences::new(records)?;
    /// seqs.remove_gap_cols();
    /// assert_eq!(seqs.record_by_id("seq1").seq(), b"AC");
    /// # Ok(()) }
    /// ```
    pub fn remove_gap_cols(&mut self) {
        assert!(
            self.aligned,
            "Cannot remove gap columns from unaligned sequences"
        );

        let mut gap_cols: BitVec = BitVec::repeat(true, self.s[0].seq().len());
        for rec in &self.s {
            let seq_gaps = rec.seq().iter().map(|&c| c == GAP).collect::<BitVec>();
            gap_cols &= seq_gaps;
        }

        let new_seqs = self.s.iter().map(|rec| {
            let seq: Vec<u8> = rec
                .seq()
                .iter()
                .enumerate()
                .filter(|(i, _)| !gap_cols[*i])
                .map(|(_, c)| *c)
                .collect();
            record!(rec.id(), rec.desc(), &seq)
        });
        self.s = new_seqs.collect();
    }

    /// Checks if all sequence IDs are unique. Runs on Sequence object creation.
    ///
    /// Returns `Ok(())` if all IDs are unique, or an error if duplicates are found.
    fn ids_are_unique(&self) -> Result<()> {
        let mut seen = HashSet::new();
        for record in self.iter() {
            let id = record.id();
            if !seen.insert(id) {
                bail!(
                    Sequence,
                    "duplicate record id ({id}) found in the sequences"
                )
            }
        }
        Ok(())
    }
}

fn detect_alphabet(sequences: &[Record]) -> &'static Alphabet {
    let dna_alphabet = Alphabet::dna();
    for record in sequences.iter() {
        if !dna_alphabet.is_word(record.seq()) {
            return Alphabet::protein();
        }
    }
    dna_alphabet
}

#[cfg(test)]
mod private_tests {
    use assert_matches::assert_matches;
    use rstest::rstest;

    use crate::{io::read_sequences, record_wo_desc as record, Error::Sequence};

    use super::*;

    #[rstest]
    #[case::aligned("./data/sequences_DNA1.fasta")]
    #[case::unaligned("./data/sequences_DNA2_unaligned.fasta")]
    #[case::long("./data/sequences_long.fasta")]
    fn dna_type_correct(#[case] input: &str) {
        let seqs = read_sequences(input).unwrap();
        let alphabet = detect_alphabet(&seqs);
        assert_eq!(alphabet, Alphabet::dna());
    }

    #[rstest]
    #[case("./data/sequences_protein1.fasta")]
    #[case("./data/sequences_protein2.fasta")]
    fn protein_type_correct(#[case] input: &str) {
        let seqs = read_sequences(input).unwrap();
        let alphabet = detect_alphabet(&seqs);
        assert_eq!(alphabet, Alphabet::protein());
    }

    #[test]
    fn ids_are_unique() {
        let result = Sequences::new(vec![
            record!("on", b"X"),
            record!("tw", b"X"),
            record!("th", b"N"),
            record!("fo", b"N"),
        ]);

        assert!(result.is_ok());
        assert!(result.unwrap().ids_are_unique().is_ok());
    }

    #[test]
    fn duplicate_ids() {
        let result = Sequences::new(vec![
            record!("on", b"X"),
            record!("tw", b"X"),
            record!("on", b"N"),
            record!("fo", b"N"),
        ]);

        assert_matches!(
            result,
            Err(Sequence(msg)) if msg.contains("duplicate record id (on) found in the sequences")
        );
    }

    #[test]
    fn duplicate_ids_from_fasta() {
        let res =
            Sequences::new(read_sequences("./data/sequences_garbage_duplicate_ids.fasta").unwrap());
        assert_matches!(
            res,
            Err(Sequence(msg)) if msg.contains("duplicate record id (C) found in the sequences")
        );
    }

    #[test]
    fn equality() {
        let mut raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        raw_seqs.reverse();
        let seqs2 = Sequences::new_unchecked(raw_seqs);
        assert_eq!(seqs1, seqs2);
    }

    #[test]
    fn inequality() {
        let mut raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        raw_seqs[1] = record!("seq2", b"CCCA");
        let seqs2 = Sequences::new_unchecked(raw_seqs);
        assert_ne!(seqs1, seqs2);
    }
    #[test]
    fn inequality_diff_lengths() {
        let mut raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        raw_seqs.pop();
        let seqs2 = Sequences::new_unchecked(raw_seqs);
        assert_ne!(seqs1, seqs2);
    }

    #[test]
    fn inequality_diff_alphabets() {
        let raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        let seqs2 = Sequences::with_alphabet_unchecked(raw_seqs, Alphabet::protein());
        assert_ne!(seqs1, seqs2);
    }

    #[test]
    fn equality_unaligned() {
        let raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCC"),
            record!("seq3", b"TTAAAAA"),
            record!("seq4", b"GGGBG"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        let seqs2 = Sequences::new_unchecked(raw_seqs);
        assert_eq!(seqs1, seqs2);
    }

    #[test]
    fn inequality_unaligned_vs_aligned() {
        let raw_seqs = vec![
            record!("seq1", b"A-C-T"),
            record!("seq2", b"C-CCC"),
            record!("seq3", b"T--AA"),
            record!("seq4", b"GGG-G"),
        ];
        let seqs1 = Sequences::new_unchecked(raw_seqs.clone());
        let seqs2 = seqs1.clone().into_gapless();
        assert_ne!(seqs1, seqs2);
    }

    #[test]
    fn record_access() {
        let raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs = Sequences::new_unchecked(raw_seqs.clone());
        for (i, rec) in raw_seqs.iter().enumerate() {
            assert_eq!(seqs[i], *rec);
        }
    }

    #[test]
    fn mut_record_access() {
        let raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let mut seqs = Sequences::new_unchecked(raw_seqs.clone());
        assert_eq!(seqs[1].seq(), b"CCCC");
        seqs[1] = record!("seq2", b"AAAA");
        assert_eq!(seqs[1].seq(), b"AAAA");
    }

    #[test]
    fn private_iterator_access() {
        let raw_seqs = vec![
            record!("seq1", b"ACGT"),
            record!("seq2", b"CCCC"),
            record!("seq3", b"TTAA"),
            record!("seq4", b"GGGG"),
        ];
        let seqs = Sequences::new_unchecked(raw_seqs.clone());
        for (i, rec) in seqs.iter().enumerate() {
            assert_eq!(raw_seqs[i], *rec);
        }
    }
}
