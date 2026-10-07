use std::cell::RefCell;
use std::fmt::Display;

use crate::alignment::Alignment;
use crate::alphabets::ParsimonySet;
use crate::likelihood::TreeSearchCost;
use crate::phylo_info::PhyloInfo;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};
use crate::Result;

#[derive(Debug, Clone)]
pub struct BasicParsimonyCost<A: Alignment> {
    info: PhyloInfo<A>,
    tmp: RefCell<BasicParsimonyInfo>,
}

impl<A: Alignment> Display for BasicParsimonyCost<A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Basic parsimony cost, match = 0.0, mismatch = 1.0")
    }
}

impl<A: Alignment> BasicParsimonyCost<A> {
    pub fn new(info: PhyloInfo<A>) -> Result<Self> {
        let tmp = RefCell::new(BasicParsimonyInfo::new(&info));
        Ok(BasicParsimonyCost { info, tmp })
    }

    fn score(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Internal(_) => self.set_internal(node_idx),
                Leaf(_) => self.set_leaf(node_idx),
            }
        }
        self.tmp.borrow().cost[usize::from(self.info.tree.root)]
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let childx_info = self.tmp.borrow().node_info[usize::from(&node.children[0])].clone();
        let childy_info = self.tmp.borrow().node_info[usize::from(&node.children[1])].clone();

        let idx = usize::from(node_idx);
        if self.tmp.borrow().node_info_valid[idx] {
            return;
        }

        let mut node_info = Vec::<ParsimonySet>::with_capacity(childx_info.len());
        let mut tmp_cost = self.tmp.borrow().cost[usize::from(node.children[0])]
            + self.tmp.borrow().cost[usize::from(node.children[1])];

        for i in 0..childx_info.len() {
            let x_set = &childx_info[i];
            let y_set = &childy_info[i];

            let set = x_set & y_set;
            if set.is_empty() {
                tmp_cost += 1.0;
                node_info.push(x_set | y_set);
            } else {
                node_info.push(set);
            }
        }

        let mut tmp = self.tmp.borrow_mut();
        tmp.cost[idx] = tmp_cost;
        if let Some(parent_idx) = node.parent {
            tmp.node_info_valid[usize::from(parent_idx)] = false;
        }

        tmp.node_info[idx] = node_info;
        tmp.node_info_valid[idx] = true;
        drop(tmp);
    }

    fn set_leaf(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let idx = usize::from(node_idx);
        if !self.tmp.borrow().node_info_valid[idx] {
            self.tmp.borrow_mut().node_info_valid[usize::from(
                node.parent
                    .expect("A leaf node should always have a parent"),
            )] = false;
            self.tmp.borrow_mut().node_info_valid[idx] = true;
        }
    }
}

impl<A: Alignment> TreeSearchCost for BasicParsimonyCost<A> {
    fn cost(&self) -> f64 {
        -self.score()
    }

    fn update_tree(&mut self, tree: Tree) {
        self.info.tree = tree;
        for idx in self.info.tree.dirty.ones() {
            self.tmp.borrow_mut().node_info_valid[idx] = false;
        }
        self.info.tree.clean();
    }

    fn tree(&self) -> &Tree {
        &self.info.tree
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BasicParsimonyInfo {
    node_info: Vec<Vec<ParsimonySet>>,
    node_info_valid: Vec<bool>,
    cost: Vec<f64>,
}

impl BasicParsimonyInfo {
    pub fn new<A: Alignment>(info: &PhyloInfo<A>) -> Self {
        let node_count = info.tree.len();
        let msa_length = info.msa.len();

        let mut node_info = vec![Vec::<ParsimonySet>::new(); node_count];
        for node in info.tree.leaves() {
            let seq = info.msa.seqs().record_by_id(&node.id).seq().to_vec();
            let leaf_map = info.msa.leaf_map(&node.idx);

            let mut leaf_seq_w_gaps: Vec<ParsimonySet> = Vec::with_capacity(msa_length);

            for &site in leaf_map.iter() {
                if let Some(pos) = site {
                    leaf_seq_w_gaps.push(info.msa.alphabet().parsimony_set(&seq[pos]).clone());
                } else {
                    leaf_seq_w_gaps.push(info.msa.alphabet().gap_set().clone());
                }
            }
            node_info[usize::from(node.idx)] = leaf_seq_w_gaps;
        }

        BasicParsimonyInfo {
            node_info,
            node_info_valid: vec![false; node_count],
            cost: vec![0.0; node_count],
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use crate::alignment::{Alignment, Sequences, MSA};
    use crate::parsimony::BasicParsimonyCost;
    use crate::phylo_info::PhyloInfo;
    use crate::{record_wo_desc as record, tree};

    #[test]
    fn repeat_basic_parsimony_score() {
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"G-GA"),
            record!("B", b"G-GG"),
            record!("C", b"AGCA"),
            record!("D", b"AGCG"),
        ]);
        let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

        let info = PhyloInfo {
            msa: MSA::from_aligned_unchecked(seqs, &tree),
            tree,
        };
        let cost = BasicParsimonyCost::new(info).unwrap();
        assert_eq!(cost.score(), 5.0);
        assert_eq!(cost.score(), 5.0);

        cost.tmp.borrow_mut().node_info_valid.fill(false);
        assert_eq!(cost.score(), 5.0);
    }
}
