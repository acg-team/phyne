/// Create a FASTA record without a description.
///
/// This macro creates a FASTA record using the re-exported `Record` type from the `bio` crate.
/// No additional imports are required.
///
/// # Examples
/// ```
/// # use phylo::record_wo_desc;
/// let record = record_wo_desc!("seq1", b"ATCG");
/// assert_eq!(record.id(), "seq1");
/// assert_eq!(record.desc(), None);
/// ```
#[macro_export]
macro_rules! record_wo_desc {
    ($id:expr, $seq:expr) => {{
        use $crate::Record;
        Record::with_attrs($id, None, $seq)
    }};
}

/// Create a FASTA record with an optional description.
///
/// This macro creates a FASTA record using the re-exported `Record` type from the `bio` crate.
/// No additional imports are required.
///
/// # Examples
/// ```
/// # use phylo::record;
/// let record = record!("seq1", Some("A sequence"), b"ATCG");
/// assert_eq!(record.id(), "seq1");
/// assert_eq!(record.desc(), Some("A sequence"));
///
/// let record = record!("seq2", None, b"TTTT");
/// assert_eq!(record.desc(), None);
/// ```
#[macro_export]
macro_rules! record {
    ($id:expr, $desc:expr, $seq:expr) => {{
        use $crate::Record;
        Record::with_attrs($id, $desc, $seq)
    }};
}

/// Create a tree from a Newick string.
/// Warning: this macro creates a tree with no checks on the Newick format.
///
/// # Examples
/// ```
/// # use phylo::tree;
/// let tree = tree!("(A,B);");
/// assert_eq!(tree.leaves().len(), 2);
/// let tree = tree!("((A,B),(C,D));");
/// assert_eq!(tree.leaves().len(), 4);
/// ```
#[macro_export]
macro_rules! tree {
    ($newick:expr) => {{
        use $crate::tree::tree_parser::from_newick;
        from_newick($newick).unwrap().pop().unwrap()
    }};
}

/// Align a sequence, returning a vector of indices for non-gap characters.
///
/// # Examples
/// ```
/// # use phylo::align;
/// let aligned = align!(b"01-2");
/// assert_eq!(aligned, vec![Some(0), Some(1), None, Some(2)]);
/// let aligned = align!(b"0--1");
/// assert_eq!(aligned, vec![Some(0), None, None, Some(1)]);
/// let aligned = align!(b"012-");
/// assert_eq!(aligned, vec![Some(0), Some(1), Some(2), None]);
/// ```
#[macro_export]
macro_rules! align {
    ($e:expr) => {{
        use $crate::alphabets::GAP;
        let mut i = 0;
        $e.iter()
            .map(|&byte| {
                if byte == GAP {
                    None
                } else {
                    i += 1;
                    Some(i - 1)
                }
            })
            .collect::<Vec<_>>()
    }};
}

/// Create an aligned sequence from a vector of option indices and a ungapped sequence.
///
/// # Examples
/// ```
/// # use phylo::aligned_seq;
/// use phylo::alphabets::GAP;
/// let indices = vec![None, Some(0), None, Some(1)];
/// let seq = b"AT";
/// let aligned = aligned_seq!(indices, seq);
/// assert_eq!(aligned, vec![GAP, b'A', GAP, b'T']);
/// ```
#[macro_export]
macro_rules! aligned_seq {
    ($vec:expr, $seq:expr) => {{
        use $crate::alphabets::GAP;
        $vec.iter()
            .map(|&opt| match opt {
                Some(i) => $seq[i],
                None => GAP,
            })
            .collect::<Vec<u8>>()
    }};
}

/// Create a parsimony site from a sequence and site flag.
///
/// **Note:** This macro is intended for internal use within this crate only.
/// The API is not guaranteed to be stable across versions.
///
/// # Examples
/// ```ignore
/// let site = site!(b"ATCG", SiteFlag::NoGap);
/// ```
#[macro_export]
#[doc(hidden)] // Hide from public documentation
macro_rules! site {
    ($s:expr, $f:expr) => {{
        use $crate::alphabets::ParsimonySet;
        use $crate::parsimony::ParsimonySite;
        ParsimonySite::new(ParsimonySet::from_slice($s), $f)
    }};
}

/// Macro for early returning errors within the crate.
///
/// This macro simplifies error handling by providing a concise way to return errors from various parts of the crate.
#[macro_export]
#[doc(hidden)] // Hide from public documentation
macro_rules! bail {
    // Usage: bail!(TreeParsing, "message", pest_error)
    (TreeParsing, $fmt:literal, $pest_err:expr $(, $arg:expr)*) => {
        return Err($crate::Error::TreeParsing(format!($fmt $(, $arg)*), $pest_err))
    };
    // Usage: bail!(TreeParsing, some_string_variable, pest_error)
    (TreeParsing, $msg:expr, $pest_err:expr) => {
        return Err($crate::Error::TreeParsing($msg.to_string(), $pest_err))
    };
    // Usage: bail!(Other, anyhow_error)
    (Other, $err:expr) => {
        return Err($crate::Error::Other($err.into()))
    };
    // Usage: bail!(SubstitutionModel, FrequencyCount, "DNA", 4, actual)
    (SubstitutionModel, FrequencyCount, $name:expr, $expected:expr, $actual:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::FrequencyCount {
                name: $name.to_string(),
                expected: $expected,
                actual: $actual,
            },
        ))
    };
    // Usage: bail!(SubstitutionModel, FrequencySum, "DNA")
    (SubstitutionModel, FrequencySum, $name:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::FrequencySum {
                name: $name.to_string(),
            },
        ))
    };
    // Usage: bail!(SubstitutionModel, NegativeFrequency, "DNA")
    (SubstitutionModel, NegativeFrequency, $name:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::NegativeFrequency {
                name: $name.to_string(),
            },
        ))
    };
    // Usage: bail!(SubstitutionModel, ParameterCount, "JC69", expected, actual)
    (SubstitutionModel, ParameterCount, $name:expr, $expected:expr, $actual:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::ParameterCount {
                name: $name.to_string(),
                expected: $expected,
                actual: $actual,
            },
        ))
    };

    // Usage: bail!(SubstitutionModel, UnequalFrequencies, "JC69")
    (SubstitutionModel, UnequalFrequencies, $name:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::UnequalFrequencies {
                name: $name.to_string(),
            },
        ))
    };

    // Usage: bail!(SubstitutionModel, message)
    (SubstitutionModel, $err:expr) => {
        return Err($crate::Error::SubstitutionModel(
            $crate::error::SubstitutionModelError::Other($err.to_string()),
        ))
    };

    // Usage: bail!(EvolutionaryModel, ParameterCount, "PIP", expected, actual)
    (EvolutionaryModel, ParameterCount, $name:expr, $expected:expr, $actual:expr) => {
        return Err($crate::Error::EvolutionaryModel(
            $crate::error::EvolutionaryModelError::ParameterCount {
                name: $name.to_string(),
                expected: $expected,
                actual: $actual,
            },
        ))
    };

    // Usage: bail!(EvolutionaryModel, message)
    (EvolutionaryModel, $err:expr) => {
        return Err($crate::Error::EvolutionaryModel(
            $crate::error::EvolutionaryModelError::Other($err.to_string()),
        ))
    };


    // Usage: bail!(Alignment, "Sequences must be aligned")
    ($variant:ident, $fmt:literal $(, $arg:expr)*) => {
        return Err($crate::Error::$variant(format!($fmt $(, $arg)*)))
    };
    // Usage: bail!(Alignment, some_string_variable)
    ($variant:ident, $err:expr) => {
        return Err($crate::Error::$variant($err.to_string()))
    };
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use std::error::Error;

    use assert_matches::assert_matches;

    use crate::error::SubstitutionModelError;
    use crate::parsimony::{ParsimonySite, SiteFlag};
    use crate::tree::Tree;
    use crate::{Error::*, Record, Result};

    #[test]
    fn test_record_macro() {
        // Test basic functionality
        let record = record!("seq1", Some("description"), b"ATCG");
        assert_eq!(record.id(), "seq1");
        assert_eq!(record.desc(), Some("description"));
        assert_eq!(record.seq(), b"ATCG");

        // Test with None description
        let record = record!("seq2", None, b"TTTT");
        assert_eq!(record.id(), "seq2");
        assert_eq!(record.desc(), None);
        assert_eq!(record.seq(), b"TTTT");

        // Test with variables
        let id = "var_seq";
        let desc = Some("variable description");
        let seq = b"GCGC";
        let record = record!(id, desc, seq);
        assert_eq!(record.id(), id);
        assert_eq!(record.desc(), desc);
        assert_eq!(record.seq(), seq);
    }

    #[test]
    fn test_record_wo_desc_macro() {
        // Test basic functionality
        let record = record_wo_desc!("seq1", b"ATCG");
        assert_eq!(record.id(), "seq1");
        assert_eq!(record.seq(), b"ATCG");
        assert_eq!(record.desc(), None);

        // Test with variables
        let id = "test_seq";
        let seq = b"AAAA";
        let record = record_wo_desc!(id, seq);
        assert_eq!(record.id(), id);
        assert_eq!(record.seq(), seq);
    }

    #[test]
    fn test_tree_macro() {
        // Test simple tree
        let tree = tree!("(A,B);");
        assert_eq!(tree.leaves().len(), 2);

        // Test more complex tree
        let tree = tree!("((A,B),(C,D));");
        assert_eq!(tree.leaves().len(), 4);
        assert_eq!(tree.len(), 7); // 4 leaves + 3 internal nodes

        // Test with variable
        let newick_str = "(seq1:0.1,seq2:0.2);";
        let tree = tree!(newick_str);
        assert_eq!(tree.leaves().len(), 2);
    }

    #[test]
    fn test_align_macro() {
        // Test basic alignment
        let alignment = align!(b"01-2");
        assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

        // Test all gaps
        let alignment = align!(b"---");
        assert_eq!(alignment, vec![None, None, None]);

        // Test no gaps
        let alignment = align!(b"012");
        assert_eq!(alignment, vec![Some(0), Some(1), Some(2)]);

        // Test with variable
        let input = b"0-1";
        let alignment = align!(input);
        assert_eq!(alignment, vec![Some(0), None, Some(1)]);
    }

    #[test]
    fn test_site_macro() {
        // Test that the macro compiles and creates sites correctly
        let site1 = site!(b"ATCG", SiteFlag::NoGap);
        let site2 = site!(b"AT", SiteFlag::GapFixed);
        let site3 = site!(b"A", SiteFlag::GapOpen);

        // Test that sites are not equal (they should have different content)
        assert_ne!(site1, site2);
        assert_ne!(site2, site3);

        // Test with variable
        let flag = SiteFlag::GapExt;
        let site = site!(b"GCGC", flag);

        // Just test that it compiles and creates a site
        // We can't easily test the internal state since fields are private
        let _ = site;
    }

    // Test macros work in different contexts
    mod nested_module {
        use crate::parsimony::SiteFlag;

        #[test]
        fn test_macros_in_nested_module() {
            // Test that macros work when used in nested modules
            let record = record!("nested", None, b"ATCG");
            assert_eq!(record.id(), "nested");

            let record = record_wo_desc!("nested", b"ATCG");
            assert_eq!(record.id(), "nested");

            let tree = tree!("(A,B);");
            assert_eq!(tree.leaves().len(), 2);

            let alignment = align!(b"01-2");
            assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

            let site = site!(b"ACGT", SiteFlag::NoGap);
            let _ = site; // Ensure it compiles
        }
    }

    #[test]
    fn test_macros_with_complex_expressions() {
        use crate::alphabets::GAP;

        // Test macros work with complex expressions as arguments
        let id = format!("seq_{}", 1);
        let record = record_wo_desc!(id.as_str(), b"ATCG");
        assert_eq!(record.id(), "seq_1");

        // Test with method calls
        let seq_vec = vec![b"ATCG"[0], b"ATCG"[1]];
        let record = record_wo_desc!("test", seq_vec.as_slice());
        assert_eq!(record.seq(), b"AT");

        // Test tree macro with string operations
        let base = "(A,B)";
        let full_tree = format!("{base};");
        let tree = tree!(full_tree.as_str());
        assert_eq!(tree.leaves().len(), 2);

        // Test alignment macro with a u8 array
        let input = [0u8, GAP, 1, 2, GAP, 3];
        let alignment = align!(input);
        assert_eq!(
            alignment,
            vec![Some(0), None, Some(1), Some(2), None, Some(3)]
        );

        // Test site macro with a variable
        let site_content = b"ACGT";
        let site = site!(site_content, SiteFlag::NoGap);
        let _ = site; // Ensure it compiles
    }

    #[test]
    fn test_macros_in_functions() {
        fn create_record(id: &str, desc: Option<&str>, seq: &[u8]) -> Record {
            record!(id, desc, seq)
        }

        let record = create_record("func_test", Some("func_desc"), b"GGGG");
        assert_eq!(record.id(), "func_test");
        assert_eq!(record.desc(), Some("func_desc"));
        assert_eq!(record.seq(), b"GGGG");

        fn create_record_wo_desc(id: &str, seq: &[u8]) -> Record {
            record_wo_desc!(id, seq)
        }

        let record = create_record_wo_desc("func_test", b"GGGG");
        assert_eq!(record.id(), "func_test");
        assert_eq!(record.desc(), None);
        assert_eq!(record.seq(), b"GGGG");

        fn create_tree(newick: &str) -> Tree {
            tree!(newick)
        }

        let tree = create_tree("(X,Y);");
        assert_eq!(tree.leaves().len(), 2);

        fn create_alignment(input: &[u8]) -> Vec<Option<usize>> {
            align!(input)
        }

        let alignment = create_alignment(b"01-2");
        assert_eq!(alignment, vec![Some(0), Some(1), None, Some(2)]);

        fn create_site(s: &[u8], f: SiteFlag) -> ParsimonySite {
            site!(s, f)
        }

        let site = create_site(b"A", SiteFlag::NoGap);
        let _ = site; // Ensure it compiles
    }

    #[test]
    fn test_macros_in_closures() {
        let records: Vec<_> = ["seq1", "seq2"]
            .iter()
            .map(|&id| record_wo_desc!(id, b"AAAA"))
            .collect();

        assert_eq!(records.len(), 2);
        assert_eq!(records[0].id(), "seq1");
        assert_eq!(records[1].id(), "seq2");

        let trees: Vec<_> = ["(A,B);", "(C,D);"]
            .iter()
            .map(|&newick| tree!(newick))
            .collect();

        assert_eq!(trees.len(), 2);
        assert_eq!(trees[0].leaves().len(), 2);
        assert_eq!(trees[1].leaves().len(), 2);

        // Test align! macro in closures
        let sequences = [b"01-2", b"0--1", b"012-"];
        let alignments: Vec<_> = sequences.iter().map(|&seq| align!(seq)).collect();

        assert_eq!(alignments.len(), 3);
        assert_eq!(alignments[0], vec![Some(0), Some(1), None, Some(2)]);
        assert_eq!(alignments[1], vec![Some(0), None, None, Some(1)]);
        assert_eq!(alignments[2], vec![Some(0), Some(1), Some(2), None]);

        // Test site! macro in closures
        let site_data = [
            (b"ATCG" as &[u8], SiteFlag::NoGap),
            (b"A", SiteFlag::GapFixed),
            (b"AG", SiteFlag::GapOpen),
        ];
        let sites: Vec<_> = site_data
            .iter()
            .map(|&(seq, flag)| site!(seq, flag))
            .collect();

        assert_eq!(sites.len(), 3);
        // Sites should be different from each other
        assert_eq!(sites[0], sites[0]);
        assert_ne!(sites[0], sites[1]);
        assert_ne!(sites[1], sites[2]);
        assert_ne!(sites[0], sites[2]);
    }

    #[test]
    #[allow(non_snake_case)] // Intentionally shadowing type names for hygiene testing
    fn test_macro_hygiene() {
        // Test that macros don't interfere with local variables
        let Record = "not_a_record"; // Shadow the Record type
        let _record = record_wo_desc!("test", b"ATCG"); // Should still work
        let _record = record!("test", Some("test_desc"), b"ATCG"); // Should still work
        assert_eq!(Record, "not_a_record"); // Local variable unchanged

        // Test with function names
        fn from_newick() -> &'static str {
            "not_a_function"
        }
        let _tree = tree!("(A,B);"); // Should still work
        assert_eq!(from_newick(), "not_a_function"); // Local function unchanged

        // Test with align macro - verify it uses the correct GAP constant
        mod alphabets {
            pub const GAP: u8 = 99; // Different GAP constant in local scope
        }
        let GAP = 100; // Local GAP constant
        let _alignment = align!(b"01-2"); // Should use $crate::alphabets::GAP, not local one
        assert_eq!(alphabets::GAP, 99); // Local mod GAP is different
        assert_eq!(GAP, 100); // Local GAP is unchanged

        // Test with site macro - shadow the actual types used in the macro
        let ParsimonySite = "not_a_site"; // Shadow the ParsimonySite type
        let ParsimonySet = "not_a_set"; // Shadow the ParsimonySet type
        let _site = site!(b"ATCG", SiteFlag::NoGap); // Should still work
        assert_eq!(ParsimonySite, "not_a_site"); // Local variable unchanged
        assert_eq!(ParsimonySet, "not_a_set"); // Local variable unchanged
    }

    #[test]
    fn bail_macro_formatting() {
        // Test bail! with format args
        fn fail_formatted() -> Result<()> {
            bail!(Io, "Formatted error: {}", 42);
        }
        assert_matches!(
            fail_formatted(),
            Err(Io(msg)) if msg == "Formatted error: 42"
        );

        // Test bail! with literal string
        fn fail_literal() -> Result<()> {
            bail!(Io, "Literal error");
        }
        assert_matches!(
            fail_literal(),
            Err(Io(msg)) if msg == "Literal error"
        );

        // Test bail! with variable
        fn fail_variable() -> Result<()> {
            let msg = "Variable error";
            bail!(Io, msg);
        }
        assert_matches!(
            fail_variable(),
            Err(Io(msg)) if msg == "Variable error"
        );
    }

    #[test]
    fn bail_subst_model_frequency_count() {
        fn fail_frequency_count() -> Result<()> {
            bail!(SubstitutionModel, FrequencyCount, "DNA", 4, 3);
        }
        assert_matches!(
            fail_frequency_count(),
            Err(SubstitutionModel(SubstitutionModelError::FrequencyCount {
                ref name,
                expected: 4,
                actual: 3,
            })) if name == "DNA"
        );
    }

    #[test]
    fn bail_subst_model_frequency_sum() {
        fn fail_frequency_sum() -> Result<()> {
            bail!(SubstitutionModel, FrequencySum, "DNA");
        }
        assert_matches!(
            fail_frequency_sum(),
            Err(SubstitutionModel(SubstitutionModelError::FrequencySum { ref name }))
                if name == "DNA"
        );
    }

    #[test]
    fn bail_subst_model_negative_frequencies() {
        fn fail_negative_frequency() -> Result<()> {
            bail!(SubstitutionModel, NegativeFrequency, "DNA");
        }
        assert_matches!(
            fail_negative_frequency(),
            Err(SubstitutionModel(SubstitutionModelError::NegativeFrequency { ref name }))
                if name == "DNA"
        );
    }

    #[test]
    fn bail_subst_model_frequency_unequal() {
        fn fail_unequal_frequencies() -> Result<()> {
            bail!(SubstitutionModel, UnequalFrequencies, "K80");
        }
        assert_matches!(
            fail_unequal_frequencies(),
            Err(SubstitutionModel(SubstitutionModelError::UnequalFrequencies { ref name }))
                if name == "K80"
        );
    }

    #[test]
    fn bail_subst_model_parameter_count() {
        fn fail_parameter_count() -> Result<()> {
            bail!(SubstitutionModel, ParameterCount, "JC69", 1, 0);
        }
        assert_matches!(
            fail_parameter_count(),
            Err(SubstitutionModel(SubstitutionModelError::ParameterCount {
                ref name,
                expected: 1,
                actual: 0,
            })) if name == "JC69"
        );
    }

    #[test]
    fn bail_subst_model_other() {
        fn fail_other() -> Result<()> {
            bail!(SubstitutionModel, "test error");
        }
        assert_matches!(fail_other(), Err(SubstitutionModel(SubstitutionModelError::Other(ref msg))) if msg == "test error");
    }

    #[test]
    fn bail_macro_in_loops() {
        fn fail_in_loop(n: usize) -> Result<()> {
            for i in 0..n {
                if i == 3 {
                    bail!(Io, "Error at i={}", i);
                }
            }
            Ok(())
        }

        assert_matches!(
            fail_in_loop(5),
            Err(Io(msg)) if msg == "Error at i=3"
        );
    }

    #[test]
    fn bail_macro_in_nested_functions() {
        fn outer_function() -> Result<()> {
            fn inner_function() -> Result<()> {
                bail!(Io, "Inner function error");
            }
            inner_function()
        }

        assert_matches!(
            outer_function(),
            Err(Io(msg)) if msg == "Inner function error"
        );
    }

    #[test]
    fn bail_macro_string_variants_display() {
        fn fail_io() -> Result<()> {
            bail!(Io, "test error");
        }
        let err = fail_io().unwrap_err();
        assert_matches!(err, Io(ref s) if s == "test error");
        assert_eq!(err.to_string(), "IO error: test error");

        fn fail_alphabet() -> Result<()> {
            bail!(Alphabet, "test error");
        }
        let err = fail_alphabet().unwrap_err();
        assert_matches!(err, Alphabet(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Alphabet error: test error");

        fn fail_sequence() -> Result<()> {
            bail!(Sequence, "test error");
        }
        let err = fail_sequence().unwrap_err();
        assert_matches!(err, Sequence(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Sequence error: test error");

        fn fail_alignment() -> Result<()> {
            bail!(Alignment, "test error");
        }
        let err = fail_alignment().unwrap_err();
        assert_matches!(err, Alignment(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Alignment error: test error");

        fn fail_ancestral() -> Result<()> {
            bail!(AncestralAlignment, "test error");
        }
        let err = fail_ancestral().unwrap_err();
        assert_matches!(err, AncestralAlignment(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Ancestral alignment error: test error");

        fn fail_tree() -> Result<()> {
            bail!(Tree, "test error");
        }
        let err = fail_tree().unwrap_err();
        assert_matches!(err, Tree(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Tree error: test error");

        fn fail_tree_move() -> Result<()> {
            bail!(TreeMove, "test error");
        }
        let err = fail_tree_move().unwrap_err();
        assert_matches!(err, TreeMove(ref s) if s == "test error");
        assert_eq!(err.to_string(), "Tree move error: test error");
    }

    #[test]
    fn bail_macro_treeparsing_display() {
        use crate::tree::tree_parser::Rule;
        use pest::error::{Error as PestError, ErrorVariant};

        fn fail_tree_parsing() -> Result<()> {
            let pest_err = PestError::<Rule>::new_from_span(
                ErrorVariant::CustomError {
                    message: String::from("pest error"),
                },
                pest::Span::new("input", 0, 1).unwrap(),
            );
            bail!(TreeParsing, "parsing error", Box::new(pest_err));
        }
        let err = fail_tree_parsing().unwrap_err();
        assert_matches!(err, TreeParsing(ref s, _) if s == "parsing error");
        assert!(err
            .to_string()
            .contains("Tree parsing error: parsing error"));
        assert!(err.to_string().contains("pest error"));
        let pest_err = err
            .source()
            .unwrap()
            .downcast_ref::<Box<PestError<Rule>>>()
            .unwrap();
        assert_matches!(pest_err.as_ref(), PestError { variant: ErrorVariant::CustomError { message }, .. } if message == "pest error");
    }

    #[test]
    fn bail_macro_other_display() {
        fn fail_other() -> Result<()> {
            bail!(Other, anyhow::anyhow!("external error"));
        }
        let err = fail_other().unwrap_err();
        assert_matches!(err, Other(ref e) if e.to_string() == "external error");
        assert_eq!(err.to_string(), "external error");
    }
}
