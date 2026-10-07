use bio::io::fasta::Record;
use hashbrown::HashMap;

use crate::alignment::{Alignment, AncestralAlignment, Sequences};
use crate::alphabets::GAP;
use crate::asr::AncestralSequenceReconstruction;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};
use crate::{aligned_seq, record};

/// Reconstructs ancestral wild card sequences, i.e. instead of the usual reconstruction that
/// returns sequences with characters of the alphabet, this just returns the presence or absence of
/// any character at a site. These are represented in an [`AncestralAlignment`] as sequences of
/// [AMB_CHAR](`crate::alphabets::AMB_CHAR`) for presence and [GAP](`crate::alphabets::GAP`)
/// for absence together with the leaf sequences.
///
/// It infers per column insertions sites to be the latest common ancestor of leaves that have a
/// character in that column, and deletion sites as nodes where there is a character
/// at the node's parent and all leaves in the subtree are gaps. The presences and absences
/// are inferred according to these indel sites.
pub struct ParsimonyPresenceAbsence {}

impl<A: Alignment, AA: AncestralAlignment> AncestralSequenceReconstruction<A, AA>
    for ParsimonyPresenceAbsence
{
    fn reconstruct_ancestral_seqs_unchecked(&self, alignment: &A, tree: &Tree) -> AA {
        let mut aligned_leaf_records = Vec::new();
        for leaf in tree.leaves() {
            let unaligned_record = alignment.seqs().record_by_id(&leaf.id);
            debug_assert_eq!(unaligned_record.id(), leaf.id);
            aligned_leaf_records.push(record!(
                &leaf.id,
                unaligned_record.desc(),
                &aligned_seq!(alignment.leaf_map(&leaf.idx), unaligned_record.seq())
            ));
        }
        aligned_leaf_records.append(&mut get_ancestral_records(tree, alignment));
        let all_seqs =
            Sequences::with_alphabet_unchecked(aligned_leaf_records, alignment.seqs().alphabet());
        AA::from_aligned_with_ancestral_unchecked(all_seqs, tree)
    }
}

// Infers ancestral sequences for every site independently:
// - insertion location is latest common ancestors of all non-gap characters
// - deletion location is node n such that every leaf in the subtree rooted in n
//   is a gap and the parent of n is not a deletion location
fn get_ancestral_records<A: Alignment>(tree: &Tree, alignment: &A) -> Vec<Record> {
    // counter[node] keeps track of the sequence index during the procedural generated mapping for the node
    let mut ancestral_seqs = HashMap::new();
    // TODO: i could also have a matrix has_char[node, site] and then
    //       loop over nodes and then sites, which might be faster due to caches
    //       see https://github.com/acg-team/rust-phylo/pull/48/files#r2046683347
    //       See issue #63 https://github.com/acg-team/rust-phylo/issues/63
    for site in 0..alignment.len() {
        // upward pass
        let has_char = non_empty_clades(alignment, tree, site);
        // downward pass
        elongate_seqs(tree, &mut ancestral_seqs, site, has_char);
    }
    let mut records = Vec::new();
    for (node_idx, seq) in ancestral_seqs {
        records.push(record!(
            tree.node_id(&node_idx),
            None,
            &seq.iter()
                .map(|&b| if b { b'X' } else { GAP })
                .collect::<Vec<u8>>()
        ));
    }
    records
}

fn elongate_seqs(
    tree: &Tree,
    ancestral_seqs: &mut HashMap<NodeIdx, Vec<bool>>,
    site: usize,
    has_char: Vec<bool>,
) {
    for node_idx in tree.preorder() {
        if let Internal(_) = node_idx {
            let children = &tree.node(node_idx).children;
            let parent = &tree.node(node_idx).parent;
            let both_chars =
                has_char[usize::from(children[0])] && has_char[usize::from(children[1])];
            let both_gap =
                !has_char[usize::from(children[0])] && !has_char[usize::from(children[1])];

            let parent_char = match parent {
                None => false,
                Some(parent) => ancestral_seqs[parent][site],
            };
            let current_char = (both_chars || parent_char) && !both_gap;
            // appending to the mapping of the node
            ancestral_seqs
                .entry(*node_idx)
                .or_insert_with(Vec::new)
                .push(current_char);
        }
    }
}

/// Returns a vector V with V[n] = true iff there is a leaf in the subtree rooted in node n that is not a gap.
fn non_empty_clades<A: Alignment>(leaf_alignment: &A, tree: &Tree, site: usize) -> Vec<bool> {
    let mut has_char = vec![false; tree.len()];
    for node_idx in tree.postorder() {
        match node_idx {
            Leaf(idx) => {
                // TODO: record_by_id is slow
                //       See issue #63 https://github.com/acg-team/rust-phylo/issues/63
                has_char[*idx] = leaf_alignment.leaf_map(node_idx)[site].is_some();
            }
            Internal(idx) => {
                let children = &tree.node(node_idx).children;
                has_char[*idx] =
                    has_char[usize::from(children[0])] || has_char[usize::from(children[1])];
            }
        }
    }
    has_char
}

#[cfg(test)]
mod tests;
