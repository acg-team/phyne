use rstest::*;

use crate::alphabets::{
    Alphabet, ParsimonySet, AMB_AMINOACIDS, AMB_CHAR, AMB_NUCLEOTIDES, AMINOACIDS, NUCLEOTIDES,
    POSSIBLE_GAPS,
};
use crate::record_wo_desc as record;

#[test]
fn alphabet_name() {
    assert_eq!(Alphabet::dna().name(), "DNA");
    assert_eq!(Alphabet::protein().name(), "protein");
}

#[test]
fn parsimony_set_iters() {
    let set = Alphabet::dna().parsimony_set_from_slice(b"ACGT");
    let from_iter = set.iter().cloned().collect::<Vec<u8>>();
    assert_eq!(from_iter.len(), 4);
    let from_into_iter = set.into_iter().collect::<Vec<u8>>();
    assert_eq!(from_into_iter.len(), 4);

    let set = Alphabet::dna().parsimony_set(&b'X').clone();
    let from_iter = set.iter().cloned().collect::<Vec<u8>>();
    assert_eq!(from_iter.len(), 4);
    let from_into_iter = set.into_iter().collect::<Vec<u8>>();
    assert_eq!(from_into_iter.len(), 4);
}

#[test]
fn dna_sets() {
    let record = record!("", b"AaCcTtGgXn-");

    let sets = record
        .seq()
        .iter()
        .map(|c| Alphabet::dna().parsimony_set(c).clone())
        .collect::<Vec<_>>();
    assert_eq!(sets.len(), 11);
    assert_eq!(sets[0], sets[1]);
    assert_eq!(sets[2], sets[3]);
    assert_eq!(sets[4], sets[5]);
    assert_eq!(sets[6], sets[7]);
    assert_eq!(sets[8], sets[9]);
    assert_eq!(sets[10], ParsimonySet::gap());
    assert_eq!(
        &(&sets[0] | &sets[2]) | &(&sets[4] | &sets[6]),
        sets[8].clone()
    );
}

#[test]
fn protein_sets() {
    let record = record!("", b"rRlLeEqQxO-");
    let sets = record
        .seq()
        .iter()
        .map(|c| Alphabet::protein().parsimony_set(c).clone())
        .collect::<Vec<_>>();
    assert_eq!(sets.len(), 11);
    assert_eq!(sets[0], sets[1]);
    assert_eq!(sets[2], sets[3]);
    assert_eq!(sets[4], sets[5]);
    assert_eq!(sets[6], sets[7]);
    assert_eq!(sets[8], sets[9]);
    assert_eq!(sets[10], ParsimonySet::gap());
}

#[test]
fn dna_characters() {
    let dna = Alphabet::dna();
    assert_eq!(dna.parsimony_set(&b'N'), dna.parsimony_set(&b'X'));
    assert_eq!(
        &(dna.parsimony_set(&b'A') | dna.parsimony_set(&b'C'))
            | &(dna.parsimony_set(&b'G') | dna.parsimony_set(&b'T')),
        *dna.parsimony_set(&b'X')
    );
    assert_eq!(dna.parsimony_set(&b'-'), &ParsimonySet::gap());
    assert!(dna
        .parsimony_set(&b'-')
        .is_disjoint(dna.parsimony_set(&b'X')));
    assert!((dna.parsimony_set(&b'-') & dna.parsimony_set(&b'X')).is_empty());
    assert_eq!(dna.parsimony_set(&b'E'), dna.parsimony_set(&b'E'));
    assert_eq!(dna.parsimony_set(&b'T'), dna.parsimony_set(&b'T'));
    assert!((dna.parsimony_set(&b'V') & dna.parsimony_set(&b'T')).is_empty());
    assert!((dna.parsimony_set(&b'D') & dna.parsimony_set(&b'C')).is_empty());
    assert!((dna.parsimony_set(&b'B') & dna.parsimony_set(&b'A')).is_empty());
    assert!((dna.parsimony_set(&b'H') & dna.parsimony_set(&b'G')).is_empty());
    assert_eq!(
        *dna.parsimony_set(&b'M'),
        dna.parsimony_set(&b'A') | dna.parsimony_set(&b'C')
    );
    assert_eq!(
        *dna.parsimony_set(&b'K'),
        dna.parsimony_set(&b'G') | dna.parsimony_set(&b'T')
    );
    assert_eq!(dna.parsimony_set(&b'-'), &ParsimonySet::gap());
}

#[test]
fn protein_characters() {
    let prot = Alphabet::protein();
    assert_eq!(prot.parsimony_set(&b'X'), prot.parsimony_set(&b'O'));
    assert_eq!(prot.parsimony_set(&b'-'), &ParsimonySet::gap());
    assert!(prot
        .parsimony_set(&b'-')
        .is_disjoint(prot.parsimony_set(&b'X')));
    assert!((prot.parsimony_set(&b'-') & prot.parsimony_set(&b'X')).is_empty());
    assert_eq!(prot.parsimony_set(&b'E'), prot.parsimony_set(&b'E'));
    assert_eq!(prot.parsimony_set(&b'N'), prot.parsimony_set(&b'N'));
    assert_eq!(
        *prot.parsimony_set(&b'B'),
        prot.parsimony_set(&b'D') | prot.parsimony_set(&b'N')
    );
    assert_eq!(
        *prot.parsimony_set(&b'Z'),
        prot.parsimony_set(&b'E') | prot.parsimony_set(&b'Q')
    );
    assert_eq!(
        *prot.parsimony_set(&b'J'),
        prot.parsimony_set(&b'I') | prot.parsimony_set(&b'L')
    );
}

#[rstest]
#[case(b"ACGT", "ACGT")]
#[case(b"TGAC", "ACGT")]
#[case(b"VG", "GV")]
#[case(b"QIFGWMNKSEYTVCPRAHLD", "ACDEFGHIKLMNPQRSTVWY")]

fn test_parsimony_set_printing(#[case] input: &[u8], #[case] output: &str) {
    let set = Alphabet::protein().parsimony_set_from_slice(input);
    assert_eq!(format!("{set}"), format!("[{}]", output));
}

#[test]
fn ambiguous_sets() {
    let dna = Alphabet::dna();
    let full_set = dna.parsimony_set(&b'X');
    assert_eq!(full_set, dna.parsimony_set(&b'N'));
    for char in NUCLEOTIDES.iter() {
        assert!(full_set.contains(char));
    }
    for char in AMB_NUCLEOTIDES.iter() {
        assert!(!(full_set & dna.parsimony_set(char)).is_empty());
    }

    let prot = Alphabet::protein();
    let full_set = prot.parsimony_set(&b'X');
    for char in AMINOACIDS.iter() {
        assert!(full_set.contains(char));
    }
    for char in AMB_AMINOACIDS.iter() {
        assert!(!(full_set & prot.parsimony_set(char)).is_empty());
    }
}

#[test]
fn dna_alphabet_display() {
    let dna = Alphabet::dna();
    let display = format!("{dna}");
    let mut lines = display.split('\n');
    let name = lines.next().unwrap();
    assert!(name.contains("DNA"));
    assert!(name.contains("length 4"));

    let valid_symbols_display = lines.next().unwrap();
    assert!(valid_symbols_display.contains("Valid symbols:"));
    for char in NUCLEOTIDES {
        assert!(valid_symbols_display.contains(&(*char as char).to_string()));
    }

    let other_symbol_display = lines.collect::<String>();
    for char in AMB_NUCLEOTIDES.iter().chain(POSSIBLE_GAPS.iter()) {
        assert!(other_symbol_display.contains(&(*char as char).to_string()));
    }
}

#[test]
fn protein_alphabet_display() {
    let prot = Alphabet::protein();
    let display = format!("{prot}");
    let mut lines = display.split('\n');
    let name = lines.next().unwrap();
    assert!(name.contains("Protein"));
    assert!(name.contains("length 20"));

    let valid_symbols_display = lines.next().unwrap();
    assert!(valid_symbols_display.contains("Valid symbols:"));
    for char in AMINOACIDS {
        assert!(valid_symbols_display.contains(&(*char as char).to_string()));
    }

    let other_symbol_display = lines.collect::<String>();
    for char in AMB_AMINOACIDS.iter().chain(POSSIBLE_GAPS.iter()) {
        assert!(other_symbol_display.contains(&(*char as char).to_string()));
    }
}

#[test]
fn dna_parsimony_sets_from_slice_all() {
    let dna = Alphabet::dna();
    let set_all = &dna.parsimony_set_from_slice(NUCLEOTIDES);
    let set_slice_x = &dna.parsimony_set_from_slice(b"X");
    let set_n = dna.parsimony_set(&b'N');
    let set_x = dna.parsimony_set(&b'X');

    assert_eq!(set_slice_x, set_x);
    assert_eq!(set_all, set_x);
    assert_eq!(set_n, set_x);
}

#[rstest]
#[case(NUCLEOTIDES, AMB_NUCLEOTIDES)]
#[case(AMINOACIDS, AMB_AMINOACIDS)]
fn parsimony_sets_from_slice_repeated(#[case] chars: &[u8], #[case] ambig_chars: &[u8]) {
    let dna = Alphabet::dna();
    let all_chars = chars
        .iter()
        .chain(ambig_chars.iter())
        .cloned()
        .collect::<Vec<u8>>();

    let set_all_repeated = &dna.parsimony_set_from_slice(&all_chars);
    let set_x = dna.parsimony_set(&AMB_CHAR);

    assert_eq!(set_all_repeated, set_x);
}

#[test]
fn protein_parsimony_sets_from_slice_all() {
    let prot = Alphabet::protein();
    let set_all = &prot.parsimony_set_from_slice(AMINOACIDS);
    let set_slice_x = &prot.parsimony_set_from_slice(b"X");
    let set_x = prot.parsimony_set(&b'X');

    assert_eq!(set_slice_x, set_x);
    assert_eq!(set_all, set_x);
}

#[rstest]
#[case(b'R', b"AG")]
#[case(b'Y', b"CT")]
#[case(b'S', b"CG")]
#[case(b'W', b"AT")]
#[case(b'K', b"GT")]
#[case(b'M', b"AC")]
#[case(b'B', b"CGT")]
#[case(b'D', b"AGT")]
#[case(b'H', b"ACT")]
#[case(b'V', b"ACG")]
#[case(b'N', b"ACGT")]
#[case(b'X', b"ACGT")]
#[case(b'Z', b"ACGT")]
fn dna_parsimony_sets_from_slice_ambigs(#[case] char_code: u8, #[case] slice: &[u8]) {
    let dna = Alphabet::dna();

    let set_char = dna.parsimony_set(&char_code);
    let set_slice = &dna.parsimony_set_from_slice(slice);

    assert_eq!(set_char, set_slice);
}

#[rstest]
#[case(b'B', b"DN")]
#[case(b'Z', b"EQ")]
#[case(b'J', b"IL")]
#[case(b'X', &AMINOACIDS)]
fn protein_parsimony_sets_from_slice_ambigs(#[case] char_code: u8, #[case] slice: &[u8]) {
    let prot = Alphabet::protein();

    let set_char = prot.parsimony_set(&char_code);
    let set_slice = &prot.parsimony_set_from_slice(slice);

    assert_eq!(set_char, set_slice);
}

#[test]
fn dna_vs_protein_parsimony_sets() {
    // Characters that could represent both a nucleotide and an amino acid, but have different parsimony sets.
    let diff_chars = b"RNDHKMSWYVBJZX";
    for char_code in diff_chars {
        let dna_set = Alphabet::dna().parsimony_set(char_code);
        let prot_set = Alphabet::protein().parsimony_set(char_code);
        assert_ne!(dna_set, prot_set);
    }
}
