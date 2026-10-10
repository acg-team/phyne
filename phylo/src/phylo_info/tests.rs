use std::path::Path;

use approx::assert_relative_eq;
use assert_matches::assert_matches;

use crate::alignment::{Alignment, AncestralAlignment, Sequences, MASA, MSA};
use crate::alphabets::{Alphabet, NUCLEOTIDES};
use crate::io::read_sequences;
use crate::phylo_info::{PhyloInfo, PhyloInfoBuilder as PIB};
use crate::{frequencies, record_wo_desc as record, tree, Error};

#[test]
fn empirical_frequencies_easy() {
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"AAAAA"),
            record!("B", b"CCCCC"),
            record!("C", b"GGGGG"),
            record!("D", b"TTTTT"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    let freqs = info.freqs();
    assert_eq!(freqs, frequencies!(&[0.25, 0.25, 0.25, 0.25]));
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn empirical_frequencies() {
    let tree = tree!("(((A:2.0,B:2.0):0.3,C:2.0):0.4,D:2.0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"TT"),
            record!("B", b"CA"),
            record!("C", b"NN"),
            record!("D", b"NN"),
        ]),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { tree, msa };
    let freqs = info.freqs();
    assert_eq!(
        freqs,
        frequencies!(&[3.0 / 8.0, 2.0 / 8.0, 2.0 / 8.0, 1.0 / 8.0])
    );
    assert_eq!(freqs.sum(), 1.0);
}

#[test]
fn setup_info_correct_unaligned() {
    let fldr = Path::new("./data");
    let res_info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(res_info.is_ok());
}

#[test]
fn setup_info_mismatched_ids_missing_tips() {
    let fldr = Path::new("./data");
    let error = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains("node IDs: [\"C\", \"D\"]"));
}

#[test]
fn setup_info_mismatched_ids_missing_sequences() {
    let fldr = Path::new("./data");
    let error = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_3.newick"),
    )
    .build();
    assert_matches!(error, Err(Error::Sequence(msg)) if msg.contains("missing sequence IDs: [\"E\", \"F\"]"));
}

#[test]
fn setup_info_missing_sequence_file() {
    let fldr = Path::new("./data");
    let seq_file = fldr.join("sequences_DNA_nonexistent.fasta");
    let error = PIB::with_attrs(&seq_file, fldr.join("tree_diff_branch_lengths_1.newick")).build();
    assert_matches!(error, Err(Error::Other(msg)) if msg.to_string().contains(&format!("Failed to read fasta from {:?}", seq_file)));
}

#[test]
fn setup_info_empty_sequence_file() {
    let fldr = Path::new("./data");
    let seq_file = fldr.join("sequences_empty.fasta");
    let error = PIB::with_attrs(&seq_file, fldr.join("tree_diff_branch_lengths_1.newick")).build();
    assert_matches!(error, Err(Error::Io(msg)) if msg.contains(&format!("no sequences found in file {}", seq_file.display())));
}

#[test]
fn setup_info_empty_tree_file() {
    let fldr = Path::new("./data");
    let error = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_empty.newick"),
    )
    .build();
    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains("no trees provided"));
}

#[test]
fn setup_info_malformed_tree_file() {
    let fldr = Path::new("./data");
    let error = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_malformed.newick"),
    )
    .build();
    assert_matches!(error, Err(Error::TreeParsing(msg, _)) if msg.contains("malformed newick string"));
}

#[test]
fn setup_info_multiple_trees() {
    let fldr = Path::new("./data");
    let res_info = PIB::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_multiple.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(res_info.tree.leaves().len(), 4);
    assert_eq!(res_info.msa.seq_count(), 4);
}

#[test]
fn setup_unaligned() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build();
    assert!(info.is_ok());
}

#[test]
fn setup_aligned_msa() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(info.msa.len(), 5);
    info.msa.seqs().into_iter().for_each(|rec| {
        assert!(!rec.seq().is_empty());
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    });
    let sequences =
        Sequences::new_unchecked(read_sequences("./data/sequences_DNA1.fasta").unwrap());
    let aligned_sequences = info.compile_alignment(None).unwrap();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn correct_setup_when_sequences_empty() {
    let fldr = Path::new("./data");
    let info = PIB::with_attrs(
        fldr.join("sequences_some_empty.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    )
    .build()
    .unwrap();
    assert_eq!(info.msa.len(), 1);
    info.msa.seqs().into_iter().for_each(|rec| {
        assert_eq!(rec.seq().to_ascii_uppercase(), rec.seq());
    });
    let sequences =
        Sequences::new_unchecked(read_sequences(fldr.join("sequences_some_empty.fasta")).unwrap());
    let aligned_sequences = info.compile_alignment(None).unwrap();
    assert_eq!(aligned_sequences, sequences);
}

#[test]
fn check_phyloinfo_creation_tree_no_seqs() {
    let info = PIB::with_attrs(
        Path::new("./data/sequences_empty.fasta"),
        Path::new("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert!(info.is_err());
}

#[test]
fn check_phyloinfo_creation_newick_mismatch_ids() {
    let info = PIB::with_attrs(
        Path::new("./data/sequences_protein1.fasta"),
        Path::new("./data/tree_diff_branch_lengths_1.newick"),
    )
    .build();
    assert!(info.is_err());
}

#[test]
fn check_empirical_frequencies() {
    let tree = tree!("((((A:2,B:2):1,C:2):1,D:2):0);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"AAAAC"),
            record!("B", b"TTTCC"),
            record!("C", b"GGGCC"),
            record!("D", b"TTAAA"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    let freqs = info.freqs();
    assert_eq!(freqs.clone().sum(), 1.0);
    assert_eq!(freqs, frequencies!(&[5.0, 5.0, 7.0, 3.0]).scale(1.0 / 20.0));
}

#[test]
fn empirical_frequencies_no_ambigs() {
    let tree = tree!("((one:2,two:2):1,(three:1,four:1):2);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("one", b"CCCCCCCC"),
            record!("two", b"AAAAAAAA"),
            record!("three", b"TTTTTTTT"),
            record!("four", b"GGGGGGGG"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_x() {
    let tree = tree!("((on:2,tw:2):1,(th:1,fo:1):2);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("on", b"XXXXXXXX"),
            record!("tw", b"XXXXXXXX"),
            record!("th", b"NNNNNNNN"),
            record!("fo", b"NNNNNNNN"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_ambig_n() {
    let tree = tree!("(((on:2,tw:2):1,th:1):4,fo:1);");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("on", b"AAAAAAAAAA"),
            record!("tw", b"XXXXXXXXXX"),
            record!("th", b"CCCCCCCCCC"),
            record!("fo", b"NNNNNNNNNN"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[0.125, 0.375, 0.375, 0.125]),
        epsilon = 1e-6
    );
}

#[test]
fn empirical_frequencies_ambig_other() {
    let tree = tree!("(A:2,B:2):1.0;");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"VVVVVVVVVV"),
            record!("B", b"TTTTVVVTVV"),
        ]),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo {
        tree: tree.clone(),
        msa,
    };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);

    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![
            record!("A", b"SSSSSSSSSSSSSSSSSSSS"),
            record!("B", b"WWWWWWWWWWWWWWWWWWWW"),
        ]),
        &tree,
    )
    .unwrap();
    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(info.freqs(), frequencies!(&[0.25; 4]), epsilon = 1e-6);
}

#[test]
fn empirical_frequencies_no_aas() {
    let tree = tree!("A:1.0;");
    let msa = MSA::from_aligned(
        Sequences::new_unchecked(vec![record!("A", b"BBBBBBBBB")]),
        &tree,
    )
    .unwrap();

    let info = PhyloInfo { tree, msa };
    assert_relative_eq!(
        info.freqs(),
        frequencies!(&[3.0 / 10.0, 3.0 / 10.0, 1.0 / 10.0, 3.0 / 10.0]),
        epsilon = 1e-6
    );
}

#[test]
fn force_protein_alphabet() {
    // If data is severely subsampled it might look like DNA even though it's really protein
    let fldr = Path::new("./data");
    let info = PIB::new(fldr.join("p226.msa.fa")).build().unwrap();
    assert_eq!(info.msa.alphabet(), Alphabet::dna());

    let fldr = Path::new("./data");
    let info = PIB::new(fldr.join("p226.msa.fa"))
        .alphabet(Some(Alphabet::protein()))
        .build()
        .unwrap();
    assert_eq!(info.msa.alphabet(), Alphabet::protein());
}

#[test]
fn build_ancestral_alignment_from_aligned_leaf_seqs_missing_record() {
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA1_missing_record.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    );
    let error = builder.build_with_ancestors();

    let expected_msg = "the number of sequences (3) does not match the number \
        of leaves (4) nor the number of nodes (7) in the tree";
    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains(expected_msg));
}

#[test]
fn build_ancestral_alignment_from_aligned_leaf_seqs() {
    // arrange
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA1.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    );

    // act
    let res_info = builder.build_with_ancestors();

    // assert
    assert!(res_info.is_ok());
    let info = res_info.unwrap();
    assert_eq!(info.msa.seq_count(), 4);
    assert_eq!(info.msa.len(), 5);
    assert_eq!(info.msa.alphabet().symbols(), NUCLEOTIDES);
    assert_eq!(info.msa.ancestral_seqs().s.len(), 3);
    assert_eq!(info.tree.len(), 7);
}

#[test]
fn build_ancestral_alignment_and_tree_from_nj() {
    // arrange
    let fldr = Path::new("./data");
    let builder = PIB::new(fldr.join("sequences_DNA1.fasta"));

    // act
    let res_info = builder.build_with_ancestors();

    // assert
    assert!(res_info.is_ok());
    let info = res_info.unwrap();
    assert_eq!(info.msa.seq_count(), 4);
    assert_eq!(info.msa.len(), 5);
    assert_eq!(info.msa.alphabet().symbols(), NUCLEOTIDES);
    assert_eq!(info.msa.ancestral_seqs().s.len(), 3);
    assert_eq!(info.tree.len(), 7);
}

#[test]
fn build_ancestral_alignment_from_aligned_leaf_seqs_mismatched_ids() {
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA1_mismatched_id.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    );

    let error = builder.build_with_ancestors();

    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains("missing node IDs: [\"Z\"]"));
}

#[test]
fn build_ancestral_alignment_from_unaligned_leaf_seqs() {
    // arrange
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    );

    // act
    let res_info = builder.build_with_ancestors();

    // assert
    assert!(res_info.is_ok());
    let info = res_info.unwrap();
    assert_eq!(info.msa.seq_count(), 4);
    assert_eq!(info.msa.ancestral_seqs().s.len(), 3);
    assert_eq!(info.tree.len(), 7);
}

#[test]
fn build_ancestral_alignment_from_unaligned_leaf_seqs_mismatched_ids() {
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned_missmatched_id.fasta"),
        fldr.join("tree_diff_branch_lengths_2.newick"),
    );

    let error = builder.build_with_ancestors();

    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains("missing node IDs: [\"Z\"]"));
}

#[test]
fn build_ancestral_alignment_from_aligned_seqs() {
    // arrange
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA1_with_ancestors.fasta"),
        fldr.join("tree_diff_branch_lengths_2_with_ancestral_ids.newick"),
    );

    // act
    let res_info = builder.build_with_ancestors();

    // assert
    assert!(res_info.is_ok());
    let info = res_info.unwrap();
    assert_eq!(info.msa.seq_count(), 4);
    assert_eq!(info.msa.ancestral_seqs().s.len(), 3);
    assert_eq!(info.tree.len(), 7);
}

#[test]
fn build_ancestral_alignment_from_aligned_seqs_mismatched_ids() {
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA1_with_ancestors_missmatched_id.fasta"),
        fldr.join("tree_diff_branch_lengths_2_with_ancestral_ids.newick"),
    );

    let error = builder.build_with_ancestors();

    assert_matches!(error, Err(Error::Tree(msg)) if msg.contains("missing node IDs: [\"Z\"]"));
}

#[test]
fn build_ancestral_alignment_from_unaligned_seqs() {
    let fldr = Path::new("./data");
    let builder = PIB::with_attrs(
        fldr.join("sequences_DNA2_unaligned_with_ancestors.fasta"),
        fldr.join("tree_diff_branch_lengths_2_with_ancestral_ids.newick"),
    );

    let error = builder.build_with_ancestors();

    let expected_msg = "building an ancestral alignment from unaligned sequences (including \
        ancestral sequences) is not supported";
    assert_matches!(error, Err(Error::AncestralAlignment(msg)) if msg.contains(expected_msg));
}

#[test]
fn masa_is_dollo() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"--GTGGA---"),
            record!("B2", b"-------NNA"),
            record!("I3", b"--T-------"),
            record!("C4", b"AGG-------"),
            record!("R5", b"--A-------"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    assert!(phylo.check_dollos_constraint().is_ok());
}

#[test]
fn masa_is_not_dollo() {
    let tree = tree!("(((A1:2.0,B2:2.0)I3:0.3,C4:2.0)R5:1.0);");
    let msa = MASA::from_aligned_with_ancestral(
        Sequences::new_unchecked(vec![
            record!("A1", b"--GTGGA---"),
            record!("B2", b"------ANNA"),
            record!("I3", b"--T---N---"),
            record!("C4", b"AGG---NN--"),
            record!("R5", b"--A-------"),
        ]),
        &tree,
    )
    .unwrap();
    let phylo = PhyloInfo { msa, tree };
    let err_msg = phylo.check_dollos_constraint().unwrap_err().to_string();
    assert!(err_msg.contains("Column 6: 2 insertions"));
    assert!(err_msg.contains("Column 7: 2 insertions"));
}
