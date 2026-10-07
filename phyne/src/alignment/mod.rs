use std::fmt::{Debug, Display};

use hashbrown::HashMap;
use itertools::Itertools;

use crate::alphabets::{Alphabet, AMB_CHAR};
use crate::asr::AncestralSequenceReconstruction;
use crate::parsimony_presence_absence::ParsimonyPresenceAbsence;
use crate::phylo_info::{
    set_missing_tree_node_ids, validate_ids_with_ancestors, validate_taxa_ids,
};
use crate::tree::{NodeIdx, NodeIdx::Internal as Int, NodeIdx::Leaf, Tree};
use crate::{align, aligned_seq, bail, record, Result};

pub mod sequences;
pub use sequences::*;
pub mod aligner;
pub use aligner::*;

/// Represents an aligned position in a sequence. Used in [`Mapping`].
pub type Position = Option<usize>;
/// Represents aligned positions of a sequence.
/// E.g. The `Mapping` for the sequence `A--T-` is `[Some(0), None, None, Some(1), None]`.
pub type Mapping = Vec<Position>;
/// For an internal node of the tree, represents the pairwise alignment of the two sub MSAs that
/// correspond to the two children of that node.
pub type InternalAlignments = HashMap<NodeIdx, PairwiseAlignment>;
pub type SeqMaps = HashMap<NodeIdx, Mapping>;

/// Represents a pairwise alignment of two sequences or MSAs. Used in [`InternalAlignments`].
#[derive(Clone, Debug, PartialEq)]
pub struct PairwiseAlignment {
    pub(crate) map_x: Mapping,
    pub(crate) map_y: Mapping,
}

impl PairwiseAlignment {
    pub fn new(map_x: Mapping, map_y: Mapping) -> PairwiseAlignment {
        assert_eq!(
            map_x.len(),
            map_y.len(),
            "Mappings must have the same length"
        );
        PairwiseAlignment { map_x, map_y }
    }

    pub fn map_x(&self) -> &Mapping {
        &self.map_x
    }

    pub fn map_y(&self) -> &Mapping {
        &self.map_y
    }

    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        debug_assert_eq!(self.map_x.len(), self.map_y.len());
        self.map_x.len()
    }
}

/// Represents an alignment of sequences that are associated with the leaves of a phylogenetic tree.
/// See also [`AncestralAlignment`].
#[allow(clippy::len_without_is_empty)]
pub trait Alignment: Display + Clone + Debug {
    /// Returns the alphabet of the sequences in the alignment.
    fn alphabet(&self) -> &Alphabet;
    /// Returns the sequences without gaps
    fn seqs(&self) -> &Sequences;
    /// Returns the length of the sequences in the alignment
    fn len(&self) -> usize;
    /// Returns the number of sequences in the alignment. Should be equal to the number of leaves
    /// in the tree.
    fn seq_count(&self) -> usize;
    fn leaf_map(&self, node: &NodeIdx) -> &Mapping;
    fn leaf_maps(&self) -> &SeqMaps;
    fn internal_alignments(&self) -> &InternalAlignments;
    /// Checks if inputs are compatible, removes columns with only gaps and calls [`Self::from_aligned_unchecked`].
    ///
    /// # Errors
    ///
    /// - bails if sequences are not aligned
    /// - bails if sequence IDs do not match the taxa IDs in the tree ([`validate_taxa_ids`])
    fn from_aligned(mut sequences: Sequences, tree: &Tree) -> Result<Self> {
        if !sequences.aligned {
            bail!(Alignment, "sequences must be aligned")
        }
        validate_taxa_ids(tree, &sequences)?;
        sequences.remove_gap_cols();
        Ok(Self::from_aligned_unchecked(sequences, tree))
    }
    /// Constructs an alignment instance from aligned sequences and a phylogenetic tree. Is called
    /// by [`Self::from_aligned`]. The caller must ensure that the sequences are aligned, that the
    /// sequence IDs are unique, and that the sequence IDs match the taxa IDs in the tree.
    fn from_aligned_unchecked(sequences: Sequences, tree: &Tree) -> Self;
}

/// Represents an alignment of sequences that are associated with all nodes of a phylogenetic tree,
/// i.e. both leaves (modern sequences) and internal nodes (ancestral sequences).
///
/// The default implementation of [`Alignment::from_aligned`] only ensures
/// prerequisites to build an alignment, not an ancestral alignment. Please overwrite this default
/// implementation and make sure to call [`Tree::node_ids_are_unique`] in addition to checks
/// your implementation requires.
// TODO: instead of having this tip here, we could change the default implementation of
// Alignment::from_aligned to ensure prerequisites for alignment as well as ancestral alignment.
pub trait AncestralAlignment: Alignment {
    fn ancestral_seqs(&self) -> &Sequences;
    fn ancestral_map(&self, node_idx: &NodeIdx) -> &Mapping;
    fn ancestral_maps(&self) -> &SeqMaps;
    fn update_ancestral_map(&mut self, node_idx: &NodeIdx, map: Mapping) -> Result<()>;
    /// Checks if inputs are compatible and calls [`Self::from_aligned_with_ancestral_unchecked`].
    /// Checks:
    /// - if sequences are aligned
    /// - if sequence IDs match the node IDs in the tree ([`validate_ids_with_ancestors`])
    /// - removes columns with only gaps ([`Sequences::remove_gap_cols`])
    ///
    /// Only overwrite this method if absolutely necessary. The default implementation
    /// ensures that prerequisites are met. Overwriting and not ensuring these checks
    /// may lead to unexpected panics or wrong results.
    fn from_aligned_with_ancestral(mut all_seqs: Sequences, tree: &Tree) -> Result<Self> {
        if !all_seqs.aligned {
            bail!(Alignment, "sequences must be aligned")
        }
        validate_ids_with_ancestors(tree, &all_seqs)?;
        all_seqs.remove_gap_cols();
        Ok(Self::from_aligned_with_ancestral_unchecked(all_seqs, tree))
    }
    /// Constructs an ancestral alignment instance from aligned sequences and a phylogenetic tree. Is called
    /// by the default implementation of [`Self::from_aligned_with_ancestral`].
    fn from_aligned_with_ancestral_unchecked(all_seqs: Sequences, tree: &Tree) -> Self;
}

#[derive(Debug, Clone)]
pub struct MSA {
    seqs: Sequences,
    leaf_maps: SeqMaps,
    internal_alignments: InternalAlignments,
    idx_to_id: Vec<String>,
}

impl Display for MSA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut aligned_records = Vec::with_capacity(self.seqs.len());
        for (node_idx, seq_map) in &self.leaf_maps {
            let id = &self.idx_to_id[usize::from(node_idx)];
            let record = self.seqs.record_by_id(id);
            let aligned_seq = aligned_seq!(seq_map, record.seq());
            aligned_records.push(record!(id, record.desc(), &aligned_seq));
        }
        write!(
            f,
            "{}",
            Sequences::with_alphabet_unchecked(aligned_records, self.seqs.alphabet())
        )
    }
}

impl MSA {
    fn stack_maps(msa_len: usize, map_x: &Mapping, map_y: &Mapping) -> Mapping {
        let mut map = Vec::with_capacity(msa_len);
        let mut ind = 0usize;
        for (x, y) in map_x.iter().zip(map_y.iter()) {
            if x.is_none() && y.is_none() {
                map.push(None);
            } else {
                map.push(Some(ind));
                ind += 1;
            }
        }
        map
    }

    fn clear_common_gaps(msa_len: usize, map_x: &Mapping, map_y: &Mapping) -> PairwiseAlignment {
        let mut upd_map_x = Vec::with_capacity(msa_len);
        let mut upd_map_y = Vec::with_capacity(msa_len);
        for (x, y) in map_x.iter().zip(map_y.iter()) {
            if x.is_some() || y.is_some() {
                upd_map_x.push(*x);
                upd_map_y.push(*y);
            }
        }
        PairwiseAlignment::new(upd_map_x, upd_map_y)
    }
}

impl Alignment for MSA {
    /// Returns the alphabet of the MSA.
    ///
    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, MSA, Sequences};
    /// use phylo::alphabets::Alphabet;
    /// use phylo::{record, tree};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::with_alphabet(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ], Alphabet::dna())?;
    /// let msa = MSA::from_aligned(seqs, &tree)?;
    /// assert_eq!(msa.alphabet(), Alphabet::dna());
    /// # Ok(()) }
    ///
    fn alphabet(&self) -> &Alphabet {
        self.seqs.alphabet
    }

    fn seqs(&self) -> &Sequences {
        &self.seqs
    }

    /// Returns the length of the MSA, i.e. the number of sites/columns.
    ///
    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, MSA, Sequences};
    /// use phylo::{record, tree};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ])?;
    /// let msa = MSA::from_aligned(seqs, &tree)?;
    /// assert_eq!(msa.len(), 4);
    /// # Ok(()) }
    /// ```
    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.leaf_maps
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    /// Returns the number of sequences in the MSA, i.e. the number of rows.
    ///
    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, MSA, Sequences};
    /// use phylo::{record, tree};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ])?;
    /// let msa = MSA::from_aligned(seqs, &tree)?;
    /// assert_eq!(msa.seq_count(), 3);
    /// # Ok(()) }
    /// ```
    fn seq_count(&self) -> usize {
        self.leaf_maps.len()
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node).unwrap()
    }

    fn leaf_maps(&self) -> &SeqMaps {
        &self.leaf_maps
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    /// Constructs an alignment instance from aligned sequences and a phylogenetic tree.
    ///
    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, MSA, Sequences};
    /// use phylo::phylo_info::PhyloInfo;
    /// use phylo::{record, tree};
    ///
    /// # use phylo::Result;
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0):1.0,C2:1.0):1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ])?;
    /// let msa = MSA::from_aligned(seqs.clone(), &tree)?;
    /// let phylo_info = PhyloInfo { msa, tree };
    /// let aligned_seqs = phylo_info.compile_alignment(None)?;
    /// assert_eq!(aligned_seqs, seqs);
    /// # Ok(()) }
    /// ```
    fn from_aligned_unchecked(seqs: Sequences, tree: &Tree) -> MSA {
        let msa_len = seqs[0].seq().len();
        let mut stack = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len());
        let mut internal_alignments = InternalAlignments::with_capacity(tree.n);
        let mut idx_to_id = vec![String::new(); tree.len()];
        for node_idx in tree.postorder() {
            match node_idx {
                Int(_) => {
                    let childs = tree.children(node_idx);
                    let map_x = stack[&childs[0]].clone();
                    let map_y = stack[&childs[1]].clone();
                    stack.insert(*node_idx, Self::stack_maps(msa_len, &map_x, &map_y));
                    internal_alignments
                        .insert(*node_idx, Self::clear_common_gaps(msa_len, &map_x, &map_y));
                }
                Leaf(_) => {
                    let seq = seqs.record_by_id(tree.node_id(node_idx)).seq();
                    stack.insert(*node_idx, align!(seq).clone());
                    idx_to_id[usize::from(node_idx)] = tree.node_id(node_idx).to_string();
                }
            }
        }
        let leaf_maps = stack
            .iter()
            .filter_map(|(idx, map)| match idx {
                Leaf(_) => Some((*idx, map.clone())),
                _ => None,
            })
            .collect();

        let seqs = seqs.into_gapless();
        MSA {
            seqs,
            leaf_maps,
            internal_alignments,
            idx_to_id,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MASA {
    leaf_seqs: Sequences,
    ancestral_seqs: Sequences,
    leaf_maps: SeqMaps,
    ancestral_maps: SeqMaps,
    // TODO: this needs to be implemented
    //       see issue #150 https://github.com/acg-team/rust-phylo/issues/150
    internal_alignments: InternalAlignments,
    idx_to_id: Vec<String>,
}

impl Display for MASA {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let both_maps = self.leaf_maps.iter().chain(self.ancestral_maps.iter());
        // sort by node id
        let both_maps = both_maps.sorted_by(|(node_idx_a, _), (node_idx_b, _)| {
            self.idx_to_id[usize::from(*node_idx_a)].cmp(&self.idx_to_id[usize::from(*node_idx_b)])
        });
        for (node_idx, seq_map) in both_maps {
            let id = &self.idx_to_id[usize::from(node_idx)];
            let record = match node_idx {
                Int(_) => self.ancestral_seqs.record_by_id(id),
                Leaf(_) => self.leaf_seqs.record_by_id(id),
            };
            let aligned_seq = aligned_seq!(seq_map, record.seq());
            write!(f, "{}", record!(id, record.desc(), &aligned_seq))?;
        }
        Ok(())
    }
}

impl Alignment for MASA {
    fn alphabet(&self) -> &Alphabet {
        self.leaf_seqs.alphabet
    }

    fn seqs(&self) -> &Sequences {
        &self.leaf_seqs
    }

    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.leaf_maps
            .values()
            .next()
            .map(|map| map.len())
            .unwrap_or(0)
    }

    fn seq_count(&self) -> usize {
        self.leaf_maps.len()
    }

    fn leaf_map(&self, node: &NodeIdx) -> &Mapping {
        self.leaf_maps.get(node).unwrap()
    }

    fn leaf_maps(&self) -> &SeqMaps {
        &self.leaf_maps
    }

    fn internal_alignments(&self) -> &InternalAlignments {
        &self.internal_alignments
    }

    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, AncestralAlignment, MASA, Sequences};
    /// use phylo::phylo_info::PhyloInfo;
    /// use phylo::{record, tree};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AAAA"),
    ///     record!("B1", Some("B1 sequence"), b"---A"),
    ///     record!("C2", Some("C2 sequence"), b"AA--"),
    /// ])?;
    /// let masa = MASA::from_aligned(seqs.clone(), &tree)?;
    /// let phylo_info = PhyloInfo { msa: masa, tree };
    /// let aligned_seqs = phylo_info.compile_alignment(None)?;
    ///
    /// // checking leaf sequences
    /// assert_eq!(aligned_seqs, seqs);
    /// // checking ancestral sequences
    /// let root_seq = phylo_info.msa.ancestral_seqs().record_by_id("I2").seq();
    /// let root_seq = String::from_utf8_lossy(root_seq);
    /// assert_eq!(root_seq, "XX");
    /// let root_map = phylo_info.msa.ancestral_map(&phylo_info.tree.root);
    /// assert_eq!(root_map, &vec![Some(0), Some(1), None, None]);
    /// // Ancestral sequences are inferred by (hard coded) ParsimonyPresenceAbsence.
    /// // Alternatively, you may call MSA::from_aligned and then call ASR on that.
    /// let i1_seq = phylo_info.msa.ancestral_seqs().record_by_id("I1").seq();
    /// let i1_seq = String::from_utf8_lossy(i1_seq);
    /// assert_eq!(i1_seq, "XXX");
    /// let i1_map = phylo_info.msa.ancestral_map(&phylo_info.tree.by_id("I1").idx);
    /// assert_eq!(i1_map, &vec![Some(0), Some(1), None, Some(2)]);
    /// // or use the align_seq macro to test seq and map at the same time
    /// # Ok(()) }
    /// ```
    fn from_aligned(sequences: Sequences, tree: &Tree) -> Result<Self> {
        let tree = &set_missing_tree_node_ids(tree)?;
        let msa = MSA::from_aligned(sequences, tree)?;
        // TODO: Do the internal_alignments, built in the line above, conform with adding ancestral seqs?
        //       see also from_aligned_with_ancestral
        //       See issue #150 https://github.com/acg-team/rust-phylo/issues/150
        // If the user wants to use a different ASR method to build the MASA, they can call
        // MSA::from_aligned and then call their desired ASR method on the MSA.
        let asr = ParsimonyPresenceAbsence {};
        asr.reconstruct_ancestral_seqs(&msa, tree)
    }

    fn from_aligned_unchecked(sequences: Sequences, tree: &Tree) -> Self {
        let msa = MSA::from_aligned_unchecked(sequences, tree);
        // TODO: do the internal alignments, built in the above line, conform with adding ancestral seqs?
        //       see also from_aligned_with_ancestral
        //       See issue #150 https://github.com/acg-team/rust-phylo/issues/150
        // If the user wants to use a different ASR method to build the MASA, they can call
        // MSA::from_aligned and then call their desired ASR method on the MSA.
        let asr = ParsimonyPresenceAbsence {};
        asr.reconstruct_ancestral_seqs_unchecked(&msa, tree)
    }
}

impl AncestralAlignment for MASA {
    fn ancestral_seqs(&self) -> &Sequences {
        &self.ancestral_seqs
    }

    fn ancestral_map(&self, node: &NodeIdx) -> &Mapping {
        self.ancestral_maps.get(node).unwrap()
    }

    fn ancestral_maps(&self) -> &SeqMaps {
        &self.ancestral_maps
    }

    /// This is needed because with the [TKF models](`crate::tkf_model::TKFModel`),
    /// we need to [re-estimate](`crate::tkf_model::reestimate::EdgeSeqsReestimator::reestimate`)
    /// the ancestral maps after a [tree move](crate::likelihood::TreeSearchCost::update_tree) was applied.
    /// Iterating through the aligned sequence (i.e., the old and new mappings), if the
    /// presence of a character is maintained the actual character from the old sequence is
    /// taken. If a character is newly introduced (i.e., was a gap before), an ambiguous
    /// character is inserted.
    fn update_ancestral_map(&mut self, node_idx: &NodeIdx, map: Mapping) -> Result<()> {
        if map.len() != self.len() {
            bail!(
                AncestralAlignment,
                "Mapping length {} does not match MSA length {}",
                map.len(),
                self.len()
            );
        }
        if self.ancestral_maps.get(node_idx).is_some() {
            self.update_ancestral_map_unchecked(node_idx, map);
            Ok(())
        } else {
            match node_idx {
                Int(_) => bail!(
                    AncestralAlignment,
                    "{node_idx} is not a valid internal node in the tree"
                ),
                Leaf(_) => bail!(
                    AncestralAlignment,
                    "ancestral map cannot be set for a leaf node like {node_idx}"
                ),
            }
        }
    }

    /// # Example
    /// ```
    /// use phylo::alignment::{Alignment, AncestralAlignment, MASA, Sequences};
    /// use phylo::phylo_info::PhyloInfo;
    /// use phylo::{record, tree};
    /// # use phylo::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("(((A0:1.0,B1:1.0)I1:1.0,C2:1.0)I2:1.0);");
    /// let seqs = Sequences::new(vec![
    ///     record!("A0", Some("A0 sequence"), b"AG-T"),
    ///     record!("B1", Some("B1 sequence"), b"---T"),
    ///     record!("C2", Some("C2 sequence"), b"AC--"),
    ///     record!("I1", Some("I1 sequence"), b"AA-A"),
    ///     record!("I2", Some("I2 sequence"), b"ACGT"),
    /// ])?;
    /// let masa = MASA::from_aligned_with_ancestral_unchecked(seqs.clone(), &tree);
    ///
    /// assert_eq!(masa.seqs().len(), 3);
    /// assert_eq!(masa.ancestral_seqs().len(), 2);
    /// # Ok(()) }
    /// ```
    fn from_aligned_with_ancestral_unchecked(all_seqs: Sequences, tree: &Tree) -> MASA {
        let mut leaf_maps = HashMap::<NodeIdx, Mapping>::with_capacity(tree.n);
        let mut ancestral_maps = HashMap::<NodeIdx, Mapping>::with_capacity(tree.len() - tree.n);
        let mut leaf_records = Vec::with_capacity(tree.n);
        let mut ancestral_records = Vec::with_capacity(tree.len() - tree.n);
        let mut idx_to_id = vec![String::new(); tree.len()];
        for node_idx in tree.postorder() {
            let record = all_seqs.record_by_id(tree.node_id(node_idx));
            let mapping = align!(record.seq());
            match node_idx {
                Int(_) => {
                    ancestral_maps.insert(*node_idx, mapping);
                    ancestral_records.push(record.clone());
                }
                Leaf(_) => {
                    leaf_maps.insert(*node_idx, mapping);
                    leaf_records.push(record.clone());
                }
            };
            idx_to_id[usize::from(node_idx)] = record.id().to_string();
        }
        let leaf_seqs = Sequences {
            s: leaf_records,
            aligned: true,
            alphabet: all_seqs.alphabet,
        };
        let leaf_seqs = leaf_seqs.into_gapless();
        let ancestral_seqs = Sequences {
            s: ancestral_records,
            aligned: true,
            alphabet: all_seqs.alphabet,
        };
        let ancestral_seqs = ancestral_seqs.into_gapless();

        // TODO: internal_alignments still missing. How do they work if there are seqs at internal nodes?
        //       see also MASA::from_aligned
        //       See issue #150 https://github.com/acg-team/rust-phylo/issues/150
        MASA {
            leaf_seqs,
            ancestral_seqs,
            leaf_maps,
            ancestral_maps,
            idx_to_id,
            internal_alignments: HashMap::<NodeIdx, PairwiseAlignment>::new(),
        }
    }
}

impl MASA {
    fn update_ancestral_map_unchecked(&mut self, node_idx: &NodeIdx, new_map: Mapping) {
        let old_map = self.ancestral_maps.get_mut(node_idx).unwrap();
        let id = &self.idx_to_id[usize::from(*node_idx)];
        let old_record = self.ancestral_seqs.record_by_id(id);
        let old_seq = old_record.seq();
        // The length of this 'new_seq' vector will be equal to the number 'of Some(_)' in 'map'
        // which indicates the presence of characters. We can therefore pre-allocate memory that is
        // needed in the maximal case and then shrink later.
        let mut new_seq = Vec::with_capacity(new_map.len());
        for (old_site, new_site) in old_map.iter_mut().zip(new_map.iter()) {
            if let Some(old_site_id) = old_site {
                // The presence of the character is maintained
                if new_site.is_some() {
                    // Character is kept
                    new_seq.push(old_seq[*old_site_id]);
                }
                // The presence of the character is not maintained, no character is added to new_seq
            }
            // A character is newly introduced
            else if new_site.is_some() {
                // Through the parameter 'map' we only know that a character is introduced, but
                // not which. Therefore, we insert an ambiguous character.
                new_seq.push(AMB_CHAR);
            }
        }
        *old_map = new_map;
        new_seq.shrink_to_fit();
        // TODO: avoid creating a new record here, see issue #143 https://github.com/acg-team/rust-phylo/issues/143
        let new_record = record!(id, old_record.desc(), &new_seq);
        self.ancestral_seqs
            .update_record(id, new_record)
            .expect("updating ancestral record failed. Please report this at https://github.com/acg-team/rust-phylo/issues");
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests;
