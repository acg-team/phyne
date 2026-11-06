use rstest::rstest;

use assert_matches::assert_matches;
use fixedbitset::FixedBitSet;
use rand::rng;
use rand::seq::IteratorRandom;

use crate::alignment::{
    mapping_to_bits, Alignment, AlignmentSimulation, AncestralAlignment, InternalAlignments,
    Mapping, PairwiseAlignment as PA, SeqMaps, Sequences, MASA, MSA,
};
use crate::alphabets::{Alphabet, AMINOACIDS, NUCLEOTIDES};
use crate::io::read_sequences;
use crate::phylo_info::PhyloInfo;
use crate::random::DefaultGenerator;
use crate::substitution_models::{dna_models::GTR, SubstModel, SubstitutionSimulator};
use crate::tree::{
    NodeIdx::{Internal as I, Leaf as L},
    Tree,
};
use crate::{align, record, record_wo_desc, tree, Error};

#[cfg(test)]
fn test_alignment(ids: &[&str]) -> Sequences {
    Sequences::new_unchecked(
        [
            record!("A0", Some("A0 sequence w 5 nucls"), b"AAAAA"),
            record!("B1", Some("B1 sequence w 1 nucl "), b"---A-"),
            record!("C2", Some("C2 sequence w 2 nucls"), b"AA---"),
            record!("D3", Some("D3 sequence w 1 nucl "), b"---A-"),
            record!("E4", Some("E4 sequence w 3 nucls"), b"-A-AA"),
        ]
        .into_iter()
        .filter(|rec| ids.contains(&rec.id()))
        .collect(),
    )
}

#[cfg(test)]
fn test_tree() -> Tree {
    tree!("((A0:1.0, B1:1.0) I5:1.0,(C2:1.0,(D3:1.0, E4:1.0) I6:1.0) I7:1.0) I8:1.0;")
}

#[cfg(test)]
fn maps() -> (InternalAlignments, SeqMaps) {
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    (
        InternalAlignments::from([
            (I(0), PA::new(align!(b"01234"), align!(b"01-23"))),
            (I(1), PA::new(align!(b"01234"), align!(b"---0-"))),
            (I(4), PA::new(align!(b"01--"), align!(b"-012"))),
            (I(6), PA::new(align!(b"-0-"), align!(b"012"))),
        ]),
        SeqMaps::from([
            (L(2), align!(aligned_seqs.s[0].seq())),
            (L(3), align!(aligned_seqs.s[1].seq())),
            (L(5), align!(aligned_seqs.s[2].seq())),
            (L(7), align!(aligned_seqs.s[3].seq())),
            (L(8), align!(aligned_seqs.s[4].seq())),
        ]),
    )
}

#[test]
fn sequences_from_aligned() {
    let seqs = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ];
    let sequences = Sequences::new_unchecked(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert!(sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(&sequences[i], rec);
    }
}

#[test]
fn sequences_from_unaligned() {
    let seqs = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ];
    let sequences = Sequences::new_unchecked(seqs.clone());
    assert_eq!(sequences.len(), 5);
    assert!(!sequences.is_empty());
    assert!(!sequences.aligned);
    for (i, rec) in seqs.iter().enumerate() {
        assert_eq!(&sequences[i], rec);
    }
}

#[test]
fn sequences_from_empty() {
    let sequences = Sequences::new_unchecked(vec![]);
    assert_eq!(sequences.len(), 0);
    assert!(sequences.is_empty());
    assert!(sequences.aligned);
}

#[test]
fn sequences_with_alphabet() {
    let records = vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ];

    let dna_seqs = Sequences::with_alphabet_unchecked(records.clone(), Alphabet::dna());
    assert_eq!(dna_seqs.alphabet().symbols(), NUCLEOTIDES);
    assert_ne!(dna_seqs.alphabet().symbols(), AMINOACIDS);
    assert_eq!(dna_seqs.alphabet(), Alphabet::dna());

    let protein_seqs = Sequences::with_alphabet_unchecked(records.clone(), Alphabet::protein());
    assert_eq!(protein_seqs.alphabet().symbols(), AMINOACIDS);
    assert_ne!(protein_seqs.alphabet().symbols(), NUCLEOTIDES);
    assert_eq!(protein_seqs.alphabet(), Alphabet::protein());
}

#[test]
fn sequences_into_gapless() {
    // arrange
    let expected_seqs = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ]);

    // act
    let gapless_seqs = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ])
    .into_gapless();

    // assert
    assert_eq!(gapless_seqs.len(), 5);
    assert!(!gapless_seqs.is_empty());
    assert!(!gapless_seqs.aligned);
    assert_eq!(gapless_seqs, expected_seqs);
}

#[test]
fn test_try_record_by_id() {
    let sequences = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
    ]);
    let rec = sequences.try_record_by_id("A0").unwrap();
    assert_eq!(rec.id(), "A0");
    assert_eq!(rec.seq(), b"AAAAAA");

    let err = sequences.try_record_by_id("ZZ");

    assert_matches!(
        err,
        Err(Error::Sequence(msg)) if msg.contains("sequence with id ZZ not found")
    );
}

#[test]
fn build_from_aligned_sequences() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let unaligned_seqs = aligned_seqs.into_gapless();
    let (node_map, leaf_map) = maps();
    let phylo = MSA::from_aligned(aligned_seqs.clone(), tree).unwrap();

    assert_eq!(*phylo.msa.internal_alignments(), node_map);
    assert_eq!(phylo.msa.leaf_maps, leaf_map);
    assert_eq!(phylo.msa.seqs, unaligned_seqs);
    assert_eq!(phylo.msa.len(), 5);
    assert_eq!(phylo.compile_alignment(None).unwrap(), aligned_seqs);
}

#[test]
fn fail_from_unaligned_sequences() {
    let seqs = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"AA"),
        record!("C2", Some("C2 sequence"), b"AAA"),
        record!("D3", Some("D3 sequence"), b"AA"),
        record!("E4", Some("E4 sequence"), b"AAAA"),
    ]);
    let tree = test_tree();
    let err = MSA::from_aligned(seqs, tree);
    assert_matches!(
        err,
        Err(Error::Alignment(msg)) if msg.contains("sequences must be aligned")
    );
}

#[test]
fn compile_msa_root() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(
        &["A0", "B1", "C2", "D3", "E4"]
            .into_iter()
            .choose_multiple(&mut rng(), 5),
    );
    let phylo = MSA::from_aligned(aligned_seqs.clone(), tree).unwrap();
    assert_eq!(phylo.compile_alignment(None).unwrap(), aligned_seqs);
}

#[test]
fn compile_msa_int1() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let node_idx = tree.idx("I5");
    let phylo = MSA::from_aligned(aligned_seqs, tree).unwrap();
    assert_eq!(
        phylo.compile_alignment(Some(&node_idx)).unwrap(),
        test_alignment(&["A0", "B1"]),
    );
}

#[test]
fn compile_msa_int2() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let d3 = test_alignment(&["D3"]).s.pop().unwrap();
    let e4 = test_alignment(&["E4"]).s.pop().unwrap();
    let node_idx = tree.idx("I6");
    let data = Sequences::new_unchecked(vec![
        record!(d3.id(), d3.desc(), b"-A-"),
        record!(e4.id(), e4.desc(), b"AAA"),
    ]);
    let phylo = MSA::from_aligned(aligned_seqs, tree).unwrap();
    assert_eq!(phylo.compile_alignment(Some(&node_idx)).unwrap(), data);
}

#[test]
fn compile_msa_leaf() {
    let tree = test_tree();
    let aligned_seqs = test_alignment(&["A0", "B1", "C2", "D3", "E4"]);
    let unaligned_seqs = aligned_seqs.clone().into_gapless();
    let phylo = MSA::from_aligned(aligned_seqs.clone(), tree).unwrap();
    for leaf_id in phylo.tree.leaf_ids() {
        assert_eq!(
            phylo
                .compile_alignment(Some(&phylo.tree.idx(&leaf_id)))
                .unwrap(),
            Sequences::new_unchecked(vec![unaligned_seqs.record_by_id(&leaf_id).clone()])
        );
    }
}

#[test]
fn input_msa_empty_col() {
    let tree = test_tree();
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_empty_col.fasta").unwrap());
    let phylo = MSA::from_aligned(sequences, tree).unwrap();
    assert_eq!(phylo.msa.len(), 40 - 1);
    assert_eq!(phylo.msa.seq_count(), 5);
}

#[test]
fn display_sequences() {
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let s = format!("{sequences}");
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();
    assert_eq!(lines, true_lines);
}

#[test]
fn display_unaligned_sequences() {
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA2_unaligned.fasta").unwrap());
    let s = format!("{sequences}");
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA2_unaligned.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();
    assert_eq!(lines, true_lines);
}

#[test]
fn fmt_alignment() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let msa = MSA::from_aligned(sequences, tree).unwrap().msa;
    let true_content = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = true_content.lines().collect::<Vec<_>>();
    true_lines.sort();

    // act
    let s = format!("{msa}");

    // assert
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);
    assert_eq!(lines, true_lines);
}

#[test]
fn display_alignment() {
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let msa = MSA::from_aligned(sequences, tree).unwrap().msa;

    let s = format!("{msa}");
    let mut lines = s.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 8);

    let s = std::fs::read_to_string("./data/sequences_DNA1.fasta").unwrap();
    let mut true_lines = s.lines().collect::<Vec<_>>();
    true_lines.sort();

    assert_eq!(lines, true_lines);
}

#[test]
fn display_ancestral_alignment() {
    // arrange
    let tree = tree!("((C:0.1,D:0.2)I01:0.3,(A:0.4,B:0.5)I02:0.6)Root;");
    let sequences = Sequences::new_unchecked(
        read_sequences("./data/sequences_DNA1_with_ancestors.fasta").unwrap(),
    );
    let msa = MASA::from_aligned_with_ancestral(sequences, tree)
        .unwrap()
        .msa;
    let true_lines = std::fs::read_to_string("./data/sequences_DNA1_with_ancestors.fasta").unwrap();
    let mut true_lines = true_lines.lines().collect::<Vec<_>>();
    true_lines.sort();

    // act
    let lines = format!("{msa}");

    // assert
    let mut lines = lines.lines().collect::<Vec<_>>();
    lines.sort();
    assert_eq!(lines.len(), 14);
    assert_eq!(lines, true_lines);
}

#[test]
fn from_aligned_with_ancestral_fails() {
    let tree = tree!("((C:0.1,D:0.2)I01:0.3,(A:0.4,B:0.5)I02:0.6)Root;");
    let seqs =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA2_unaligned.fasta").unwrap());

    let err = MASA::from_aligned_with_ancestral(seqs, tree);

    assert_matches!(
        err,
        Err(Error::Alignment(msg)) if msg.contains("must be aligned")
    );
}

#[test]
fn masa_compile_root() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let phylo = MASA::from_aligned_unchecked(sequences.clone(), tree);

    // act
    let compiled = phylo.compile_alignment(None).unwrap();

    // assert
    assert_eq!(compiled, sequences);
}

#[test]
fn masa_compile_subroot() {
    // arrange
    let tree = tree!("(C:0.06465432,D:27.43128366,(A:0.00000001,B:0.00000001):0.08716381);");
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let subroot_id = "C";
    let phylo = MASA::from_aligned(sequences.clone(), tree).unwrap();
    let mut true_compiled = Sequences::new_unchecked(
        sequences
            .into_iter()
            .filter(|f| f.id() == subroot_id)
            .cloned()
            .collect(),
    );
    true_compiled.remove_gap_cols();

    // act
    let compiled = phylo
        .compile_alignment(Some(&phylo.tree.idx(subroot_id)))
        .unwrap();

    // assert
    assert_eq!(compiled, true_compiled);
}

#[test]
fn removing_gap_cols() {
    let mut seqs = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"AAAAAA"),
        record!("B1", Some("B1 sequence"), b"---A-A"),
        record!("C2", Some("C2 sequence"), b"AA---A"),
        record!("D3", Some("D3 sequence"), b"---A-A"),
        record!("E4", Some("E4 sequence"), b"-A-AAA"),
    ]);
    seqs.remove_gap_cols();
    assert_eq!(seqs.len(), 5);
    for seq in &seqs {
        assert_eq!(seq.seq().len(), 6);
    }

    let mut seqs2 = Sequences::new_unchecked(vec![
        record!("A0", Some("A0 sequence"), b"-AAA-AA-A"),
        record!("B1", Some("B1 sequence"), b"-----A--A"),
        record!("C2", Some("C2 sequence"), b"-AA-----A"),
        record!("D3", Some("D3 sequence"), b"-----A--A"),
        record!("E4", Some("E4 sequence"), b"--A--AA-A"),
    ]);
    for seq in &seqs2 {
        assert_eq!(seq.seq().len(), 9);
    }

    seqs2.remove_gap_cols();
    assert_eq!(seqs2.len(), 5);
    for seq in &seqs2 {
        assert_eq!(seq.seq().len(), 6);
    }
    assert_eq!(seqs, seqs2)
}

#[test]
#[should_panic]
fn removing_gap_cols_on_unaligned() {
    let mut seqs = Sequences::new_unchecked(vec![
        record!("A0", None, b"AAAAAA"),
        record!("B1", None, b"AA"),
        record!("C2", None, b"AAA"),
        record!("D3", None, b"AA"),
        record!("E4", None, b"AAAA"),
    ]);
    seqs.remove_gap_cols();
}

#[test]
fn sequence_iterator_access() {
    let records = vec![record!("seq1", None, b"A"), record!("seq2", None, b"C")];
    let seqs = Sequences::new_unchecked(records);
    let mut iter = seqs.into_iter();
    assert_eq!(iter.next().unwrap().id(), "seq1");
    let second = iter.next();
    assert!(second.is_some());
    assert_eq!(second.unwrap().id(), "seq2");
}

#[test]
fn update_ancestral_map_valid() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let node_id = "I3";
    let node_idx = tree.idx(node_id);
    let mut info = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record_wo_desc!("A1", b"--GTGGA---"),
            record_wo_desc!("B2", b"-------NNA"),
            record_wo_desc!("I3", b"--T-GC----"),
            record_wo_desc!("C4", b"AGG-------"),
            record_wo_desc!("R5", b"--A-------"),
        ]),
        tree,
    )
    .unwrap();

    assert_eq!(info.msa.ancestral_map(&node_idx), &align!(b"--N-NN----"));
    // the actual chars are not important in the creation of the new_map, just the presence/absence of gaps
    let new_map = align!(b"A-AA-AAA-A");
    info.msa
        .update_ancestral_map(&node_idx, new_map.clone())
        .expect("should update ancestral map successfully");
    assert_eq!(info.msa.ancestral_map(&node_idx), &new_map);
    assert_eq!(
        info.msa.ancestral_seqs().record_by_id(node_id).seq(),
        b"XTXCXXX"
    );

    let new_map = align!(b"---A-A---A");
    info.msa
        .update_ancestral_map(&node_idx, new_map.clone())
        .expect("should update ancestral map successfully");
    assert_eq!(info.msa.ancestral_map(&node_idx), &new_map);
    assert_eq!(
        info.msa.ancestral_seqs().record_by_id(node_id).seq(),
        b"XCX"
    );

    let new_map = align!(b"-----A--A-");
    info.msa
        .update_ancestral_map(&node_idx, new_map.clone())
        .expect("should update ancestral map successfully");
    assert_eq!(info.msa.ancestral_map(&node_idx), &new_map);
    assert_eq!(info.msa.ancestral_seqs().record_by_id(node_id).seq(), b"CX");
}

#[test]
fn update_ancestral_maps_fails_wrong_length() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let node_id = "I3";
    let node_idx = tree.idx(node_id);
    let mut info = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record_wo_desc!("A1", b"--GTGGA---"),
            record_wo_desc!("B2", b"-------NNA"),
            record_wo_desc!("I3", b"--T-GC----"),
            record_wo_desc!("C4", b"AGG-------"),
            record_wo_desc!("R5", b"--A-------"),
        ]),
        tree,
    )
    .unwrap();
    let new_map = align!(b"A-AAAA-A");
    let err = info.msa.update_ancestral_map(&node_idx, new_map);
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("does not match MSA length")
    );
}

#[test]
fn update_ancestral_map_nonexistent_internal() {
    let tree = tree!("((C:0.1,D:0.2)I01:0.3,(A:0.4,B:0.5)I02:0.6)Root;");
    let sequences = Sequences::new_unchecked(
        read_sequences("./data/sequences_DNA1_with_ancestors.fasta").unwrap(),
    );
    let mut info = MASA::from_aligned_with_ancestral(sequences, tree).unwrap();
    let err = info.msa.update_ancestral_map(&I(10), align!(b"-ACGT"));
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("node 10 is not a valid internal node in the tree")
    );
}

#[test]
fn update_ancestral_map_leaf() {
    let tree = tree!("((C:0.1,D:0.2)I01:0.3,(A:0.4,B:0.5)I02:0.6)Root;");
    let sequences = Sequences::new_unchecked(
        read_sequences("./data/sequences_DNA1_with_ancestors.fasta").unwrap(),
    );
    let mut info = MASA::from_aligned_with_ancestral(sequences, tree).unwrap();
    let err = info.msa.update_ancestral_map(&L(2), align!(b"-ACGT"));
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("ancestral map cannot be set for a leaf node")
    );
}

#[test]
fn remove_extinct_columns_masa() {
    let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");

    let records = vec![
        record_wo_desc!("A", b"A---T--"),
        record_wo_desc!("B", b"C--CT--"),
        record_wo_desc!("C", b"G--CT-G"),
        record_wo_desc!("I", b"T--CTAG"),
        record_wo_desc!("R", b"A-AATGG"),
    ];

    let seqs = Sequences::new(records).unwrap();

    for rec in &seqs.s {
        assert_eq!(rec.seq().len(), 7, "sequence length mismatch");
    }

    let mut phylo = MASA::from_aligned_with_ancestral(seqs, tree).unwrap();

    // sanity checks before removal (the second column is removed when the MASA is created)
    assert_eq!(phylo.msa.len(), 6, "lens differ for initial length");
    // after removing the two extinct columns the length should be 4
    phylo.msa.remove_extinct_columns();
    assert_eq!(phylo.msa.len(), 4, "lens differ after removal");

    let PhyloInfo { msa: masa, tree } = phylo;

    let a = &tree.idx("A");
    assert_eq!(masa.leaf_maps().get(a).unwrap(), &align!(b"N-N-"), "A");
    let b = &tree.idx("B");
    assert_eq!(masa.leaf_maps().get(b).unwrap(), &align!(b"NNN-"), "B");
    let c = &tree.idx("C");
    assert_eq!(masa.leaf_maps().get(c).unwrap(), &align!(b"NNNN"), "C");
    let i = &tree.idx("I");
    assert_eq!(masa.ancestral_maps().get(i).unwrap(), &align!(b"NNNN"), "I");
    let r = &tree.idx("R");
    assert_eq!(masa.ancestral_maps().get(r).unwrap(), &align!(b"NNNN"), "R");

    assert_eq!(masa.seqs().record_by_id("A").seq(), b"AT");
    assert_eq!(masa.seqs().record_by_id("B").seq(), b"CCT");
    assert_eq!(masa.seqs().record_by_id("C").seq(), b"GCTG");
    assert_eq!(masa.ancestral_seqs().record_by_id("I").seq(), b"TCTG");
    assert_eq!(masa.ancestral_seqs().record_by_id("R").seq(), b"AATG");
}

#[test]
fn remove_extinct_columns_masa_no_extinction() {
    let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");

    // every column has at least one character in a leaf, so nothing goes extinct
    let records = vec![
        record_wo_desc!("A", b"A--T"),
        record_wo_desc!("B", b"AC-T"),
        record_wo_desc!("C", b"ACGT"),
        record_wo_desc!("I", b"ACGT"),
        record_wo_desc!("R", b"ACGT"),
    ];

    let mut phylo =
        MASA::from_aligned_with_ancestral(Sequences::new(records).unwrap(), tree).unwrap();

    // no column is all-gap in the leaves, so the lengths must be unchanged
    assert_eq!(phylo.msa.len(), 4, "lens differ for initial length");
    phylo.msa.remove_extinct_columns();
    assert_eq!(phylo.msa.len(), 4, "lens differ after removal");

    let PhyloInfo { msa, tree } = phylo;

    // mappings are untouched, only the gap/non-gap pattern matters
    let a = &tree.idx("A");
    assert_eq!(msa.leaf_maps().get(a).unwrap(), &align!(b"A--T"), "A");
    let b = &tree.idx("B");
    assert_eq!(msa.leaf_maps().get(b).unwrap(), &align!(b"AC-T"), "B");
    let c = &tree.idx("C");
    assert_eq!(msa.leaf_maps().get(c).unwrap(), &align!(b"ACGT"), "C");
    let i = &tree.idx("I");
    assert_eq!(msa.ancestral_maps().get(i).unwrap(), &align!(b"ACGT"), "I");
    let r = &tree.idx("R");
    assert_eq!(msa.ancestral_maps().get(r).unwrap(), &align!(b"ACGT"), "R");

    assert_eq!(msa.seqs().record_by_id("A").seq(), b"AT");
    assert_eq!(msa.seqs().record_by_id("B").seq(), b"ACT");
    assert_eq!(msa.seqs().record_by_id("C").seq(), b"ACGT");
    assert_eq!(msa.ancestral_seqs().record_by_id("I").seq(), b"ACGT");
    assert_eq!(msa.ancestral_seqs().record_by_id("R").seq(), b"ACGT");
}

#[test]
fn remove_extinct_columns_masa_everything_dies() {
    let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");

    // every column has a character in the ancestral nodes but only gaps in the leaves,
    // so all columns are extinct in the leaves
    let records = vec![
        record_wo_desc!("A", b"----"),
        record_wo_desc!("B", b"----"),
        record_wo_desc!("C", b"----"),
        record_wo_desc!("I", b"A-GT"),
        record_wo_desc!("R", b"ACGT"),
    ];

    let mut phylo =
        MASA::from_aligned_with_ancestral(Sequences::new(records).unwrap(), tree).unwrap();

    // the columns survive creation because the ancestral nodes have characters
    assert_eq!(phylo.msa.len(), 4, "lens differ for initial length");
    // all columns go extinct in the leaves, so the alignment becomes empty
    phylo.msa.remove_extinct_columns();
    assert_eq!(phylo.msa.len(), 0, "lens differ after removal");

    let PhyloInfo { msa, tree } = phylo;

    for id in ["A", "B", "C"] {
        assert!(
            msa.leaf_maps().get(&tree.idx(id)).unwrap().is_empty(),
            "{id}"
        );
        assert!(
            msa.seqs().record_by_id(id).seq().is_empty(),
            "{id} sequence should be empty"
        );
    }
    for id in ["I", "R"] {
        assert!(
            msa.ancestral_maps().get(&tree.idx(id)).unwrap().is_empty(),
            "{id}"
        );
        assert!(
            msa.ancestral_seqs().record_by_id(id).seq().is_empty(),
            "{id} sequence should be empty"
        );
    }
}

#[test]
fn into_alignment_masa_to_msa() {
    let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R:1.0;");
    let records = vec![
        record_wo_desc!("A", b"A---C-T"),
        record_wo_desc!("B", b"---CG-T"),
        record_wo_desc!("C", b"A-----T"),
        record_wo_desc!("I", b"A-ACG-T"),
        record_wo_desc!("R", b"A-ACG-T"),
    ];
    let seqs = Sequences::new(records).unwrap();
    let phylo = MASA::from_aligned_with_ancestral(seqs, tree).unwrap();

    // columns 2 and 5 are only gaps, so they are removed during the creation
    assert_eq!(phylo.msa.len(), 5);

    let phylo: PhyloInfo<MSA> = phylo.discard_ancestral_seqs();

    assert_eq!(phylo.msa.seq_count(), 3);
    // column 3 is only gaps in the leaf sequences, so it is removed during the transformation
    assert_eq!(phylo.msa.len(), 4);

    assert_eq!(phylo.msa.seqs().record_by_id("A").seq(), b"ACT");
    assert_eq!(phylo.msa.seqs().record_by_id("B").seq(), b"CGT");
    assert_eq!(phylo.msa.seqs().record_by_id("C").seq(), b"AT");

    let a_idx = phylo.tree.idx("A");
    let b_idx = phylo.tree.idx("B");
    let c_idx = phylo.tree.idx("C");

    assert_eq!(phylo.msa.leaf_map(&a_idx), &align!(b"A-CT"));
    assert_eq!(phylo.msa.leaf_map(&b_idx), &align!(b"-CGT"));
    assert_eq!(phylo.msa.leaf_map(&c_idx), &align!(b"A--T"));
}

#[test]
fn simulate_alignment_matches_discarded_ancestral_alignment() {
    let tree = tree!("((A:2.0,B:2.0)I:1.0,C:1.0)R:1.0;");
    let model = SubstModel::<GTR>::default();
    let msa_len = 20;
    let simulator = SubstitutionSimulator::new(model, tree, msa_len).unwrap();

    // the default `simulate_alignment` must equal simulating a MASA and discarding ancestors
    let seed = 123;
    let mut rng1 = DefaultGenerator::new(seed);
    let leaf_only: PhyloInfo<MSA> = simulator.simulate_alignment(&mut rng1);

    let mut rng2 = DefaultGenerator::new(seed);
    let from_ancestral: PhyloInfo<MSA> = simulator
        .simulate_ancestral_alignment::<MASA>(&mut rng2)
        .discard_ancestral_seqs();

    assert_eq!(leaf_only.msa.len(), from_ancestral.msa.len());
    assert_eq!(leaf_only.msa.seq_count(), from_ancestral.msa.seq_count());
    for id in ["A", "B", "C"] {
        assert_eq!(
            leaf_only.msa.seqs().record_by_id(id).seq(),
            from_ancestral.msa.seqs().record_by_id(id).seq(),
            "{id}"
        );
    }
}

#[rstest]
#[case(align!(b"-0-1--2-"), &[1, 3, 6])]
#[case(align!(b"-----------"), &[])]
#[case(align!(b"01"), &[0, 1])]
#[case(align!(b"0123456789"), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])]
#[case(align!(b"----------01234"), &[10, 11, 12, 13, 14])]
#[case(align!(b"0---------1"), &[0, 10])]
fn test_mapping_to_bits(#[case] mapping: Mapping, #[case] expected: &[usize]) {
    let bits = mapping_to_bits(&mapping);

    let expected = FixedBitSet::from_iter(expected.iter().copied());
    assert!(
        bits.ones().eq(expected.ones()),
        "Mapping did not match expected bits: expected: {expected} got: {bits}"
    );
}

fn test_phylo_from_aligned() -> PhyloInfo<MSA> {
    let tree = tree!("((A2:1,B3:1)I1:1,(C5:1,D6:1)I4:1)I0:0;");
    let sequences = Sequences::with_alphabet_unchecked(
        vec![
            record_wo_desc!("A2", b"XXX--X"),
            record_wo_desc!("B3", b"X-X--X"),
            record_wo_desc!("C5", b"X---X-"),
            record_wo_desc!("D6", b"--XXX-"),
        ],
        Alphabet::dna(),
    );
    MSA::from_aligned(sequences, tree.clone()).unwrap()
}

#[test]
fn masa_from_msa_and_ancestral() {
    let phylo = test_phylo_from_aligned();
    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I0", b"X-X---"),
        record_wo_desc!("I1", b"X-X--X"),
        record_wo_desc!("I4", b"X-X-X-"),
    ]);

    let ancestral_gapless = ancestral_seqs.clone().into_gapless();
    let ancestral_maps = ancestral_seqs
        .into_iter()
        .map(|r| (phylo.tree.idx(r.id()), align!(r.seq())))
        .collect::<SeqMaps>();

    let phylo_w_ancestors = MASA::from_msa_with_ancestral(phylo.clone(), ancestral_seqs).unwrap();

    assert_eq!(phylo_w_ancestors.msa.leaf_maps(), phylo.msa.leaf_maps());
    assert_eq!(phylo_w_ancestors.msa.seqs(), phylo.msa.seqs());
    assert_eq!(phylo_w_ancestors.msa.ancestral_seqs(), &ancestral_gapless);
    assert_eq!(phylo_w_ancestors.msa.ancestral_maps(), &ancestral_maps);
}

#[test]
fn masa_from_msa_and_ancestral_unaligned_error() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I0", b"XX"),
        record_wo_desc!("I1", b"XXX"),
        record_wo_desc!("I4", b"XXX"),
    ]);

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("sequences must be aligned")
    );
}

#[test]
fn masa_from_msa_and_ancestral_wrong_length_error() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I0", b"X-X"),
        record_wo_desc!("I1", b"XXX"),
        record_wo_desc!("I4", b"XXX"),
    ]);

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("sequences must have the same length as the provided alignment")
    );
}

#[test]
fn masa_from_msa_with_ancestral_wrong_alphabet() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::with_alphabet_unchecked(
        vec![
            record_wo_desc!("I0", b"X-X---"),
            record_wo_desc!("I1", b"X-X--X"),
            record_wo_desc!("I4", b"X-X-X-"),
        ],
        Alphabet::protein(),
    );

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::AncestralAlignment(msg)) if msg.contains("ancestral sequences must have the same alphabet as the provided alignment")
    );
}

#[test]
fn masa_from_msa_with_ancestral_wrong_seq_ids() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I5", b"X-X---"),
        record_wo_desc!("I1", b"X-X--X"),
        record_wo_desc!("I4", b"X-X-X-"),
    ]);

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::Tree(msg))
            if msg.contains("mismatched sequence and tree internal node IDs found, missing node IDs: [\"I5\"]")
    );
}

#[test]
fn masa_from_msa_with_ancestral_extra_ids() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I0", b"X-X---"),
        record_wo_desc!("I1", b"X-X--X"),
        record_wo_desc!("I4", b"X-X-X-"),
        record_wo_desc!("ADD4", b"X-X-X-"),
    ]);

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::Tree(msg))
            if msg.contains("mismatched sequence and tree internal node IDs found, missing node IDs: [\"ADD4\"]")
    );
}

#[test]
fn masa_from_msa_with_ancestral_missing_seqs() {
    let phylo = test_phylo_from_aligned();

    let ancestral_seqs = Sequences::new_unchecked(vec![
        record_wo_desc!("I0", b"X-X---"),
        record_wo_desc!("I1", b"X-X--X"),
    ]);

    let err = MASA::from_msa_with_ancestral(phylo, ancestral_seqs);
    assert_matches!(
        err,
        Err(Error::Sequence(msg))
            if msg.contains("mismatched sequence and tree internal node IDs found, missing sequence IDs: [\"I4\"]")
    );
}
