use std::cell::RefCell;
use std::fmt::Display;

use fixedbitset::FixedBitSet;
use hashbrown::HashSet;

use crate::alignment::Alignment;
use crate::alphabets::ParsimonySet;
use crate::likelihood::TreeSearchCost;
use crate::parsimony::{
    scoring::{GapCost, SimpleScoring},
    ParsimonyScoring,
};
use crate::phylo_info::PhyloInfo;
use crate::tree::{
    NodeIdx::{self, Internal, Leaf},
    Tree,
};

#[derive(Debug, Clone)]
pub struct DolloParsimonyCost<S: ParsimonyScoring, A: Alignment> {
    pub(crate) info: PhyloInfo<A>,
    tmp: RefCell<DolloParsimonyInfo>,
    scoring: S,
}

impl<S: ParsimonyScoring, A: Alignment> Display for DolloParsimonyCost<S, A> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Dollo parsimony using: \n\t{}", self.scoring)
    }
}

impl<A: Alignment> DolloParsimonyCost<SimpleScoring, A> {
    pub fn new(info: PhyloInfo<A>) -> Self {
        let tmp = RefCell::new(DolloParsimonyInfo::new(&info));
        DolloParsimonyCost {
            info,
            tmp,
            scoring: SimpleScoring::new(1.0, GapCost::new(1.0, 1.0)),
        }
    }
}
impl<S: ParsimonyScoring, A: Alignment> DolloParsimonyCost<S, A> {
    pub fn with_scoring(info: PhyloInfo<A>, scoring: S) -> Self {
        let tmp = RefCell::new(DolloParsimonyInfo::new(&info));
        DolloParsimonyCost { info, tmp, scoring }
    }

    fn score(&self) -> f64 {
        for node_idx in self.info.tree.postorder() {
            match node_idx {
                Leaf(_) => {
                    self.set_leaf(node_idx);
                }
                Internal(_) => {
                    self.set_internal(node_idx);
                }
            }
        }
        self.tmp.borrow().cost[usize::from(self.info.tree.root)]
    }

    fn set_internal(&self, node_idx: &NodeIdx) {
        let node = self.info.tree.node(node_idx);
        let idx = usize::from(node_idx);

        if self.tmp.borrow().node_info_valid[idx] {
            return;
        }

        let childx_idx = usize::from(&node.children[0]);
        let childy_idx = usize::from(&node.children[1]);
        let blen = node.blen;

        let mut tmp = self.tmp.borrow_mut();
        for (site_idx, (x, y)) in tmp.node_leaf_sets[childx_idx]
            .clone()
            .into_iter()
            .zip(tmp.node_leaf_sets[childy_idx].clone())
            .enumerate()
        {
            tmp.node_leaf_sets[idx][site_idx] = x.union(&y).cloned().collect();
            if (tmp.mrca_reached[childx_idx][site_idx] | tmp.mrca_reached[childy_idx][site_idx])
                || tmp.node_leaf_sets[idx][site_idx] == tmp.leaf_sets[site_idx]
            {
                tmp.mrca_reached[idx].set(site_idx, true);
            }
        }

        let childx_sets = &tmp.node_info[childx_idx].clone();
        let childy_sets = &tmp.node_info[childy_idx].clone();
        let mut cost = tmp.cost[childx_idx] + tmp.cost[childy_idx];

        let gap = self.info.msa.alphabet().gap_set();
        let node_info: Vec<ParsimonySet> = childx_sets
            .iter()
            .zip(childy_sets)
            .enumerate()
            .map(|(i, (x_set, y_set))| {
                let intersection = x_set & y_set;
                if !intersection.is_empty() {
                    if intersection != *gap {
                        cost += self.scoring.min_match(blen, &intersection, &intersection);
                    }
                    intersection
                } else if tmp.mrca_reached[childx_idx][i] | tmp.mrca_reached[childy_idx][i] {
                    gap.clone()
                } else {
                    if x_set == gap || y_set == gap {
                        cost += self.scoring.gap_open(blen);
                    } else {
                        cost += self.scoring.min_match(blen, x_set, y_set);
                    }
                    &(x_set | y_set) - gap
                }
            })
            .collect();

        tmp.cost[idx] = cost;
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

        if self.tmp.borrow().node_info_valid[idx] {
            return;
        }

        let mut tmp = self.tmp.borrow_mut();

        tmp.node_info_valid[usize::from(
            node.parent
                .expect("A leaf node should always have a parent"),
        )] = false;

        tmp.node_info_valid[idx] = true;
        drop(tmp);
    }
}

impl<S: ParsimonyScoring, A: Alignment> TreeSearchCost for DolloParsimonyCost<S, A> {
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

    fn blen_optimisation(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DolloParsimonyInfo {
    node_info: Vec<Vec<ParsimonySet>>,
    node_info_valid: Vec<bool>,
    mrca_reached: Vec<FixedBitSet>,
    leaf_sets: Vec<HashSet<usize>>,
    node_leaf_sets: Vec<Vec<HashSet<usize>>>,
    cost: Vec<f64>,
}

impl DolloParsimonyInfo {
    pub fn new<A: Alignment>(info: &PhyloInfo<A>) -> Self {
        let node_count = info.tree.len();
        let leaf_count = info.tree.leaves().len();
        let msa_length = info.msa.len();

        let mut node_info = vec![vec![ParsimonySet::empty(); msa_length]; node_count];
        let mut mrca_reached = vec![FixedBitSet::with_capacity(msa_length); node_count];

        let mut leaf_sets = vec![HashSet::with_capacity(leaf_count); msa_length];
        let mut node_leaf_sets =
            vec![vec![HashSet::with_capacity(leaf_count); msa_length]; node_count];

        for leaf in info.tree.leaves() {
            let seq = info.msa.seqs().record_by_id(&leaf.id).seq().to_vec();
            let leaf_map = info.msa.leaf_map(&leaf.idx);
            let node_idx = usize::from(leaf.idx);

            mrca_reached[node_idx].set_range(.., false);

            for (idx, &site) in leaf_map.iter().enumerate() {
                if let Some(pos) = site {
                    node_info[node_idx][idx] = info.msa.alphabet().parsimony_set(&seq[pos]).clone();
                    leaf_sets[idx].insert(node_idx);
                    node_leaf_sets[node_idx][idx].insert(node_idx);
                } else {
                    node_info[node_idx][idx] = info.msa.alphabet().gap_set().clone();
                }
            }
        }

        for leaf in info.tree.leaves() {
            let leaf_idx = usize::from(leaf.idx);
            for (site, leaf_set) in leaf_sets.iter().enumerate() {
                if node_leaf_sets[leaf_idx][site] == *leaf_set {
                    mrca_reached[leaf_idx].set(site, true);
                }
            }
        }

        DolloParsimonyInfo {
            node_info,
            node_info_valid: vec![false; node_count],
            mrca_reached,
            leaf_sets,
            node_leaf_sets,
            cost: vec![0.0; node_count],
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use crate::alignment::{Alignment, Sequences, MSA};
    use crate::alphabets::Alphabet;
    use crate::likelihood::TreeSearchCost;
    use crate::parsimony::{DolloParsimonyCost, GapCost, SimpleScoring};
    use crate::phylo_info::PhyloInfo;
    use crate::{record_wo_desc as record, tree};

    #[test]
    fn repeat_dollo_parsimony_score() {
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"G-GA"),
            record!("B", b"G-GG"),
            record!("C", b"AGCA"),
            record!("D", b"AGCG"),
        ]);

        let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

        let info = PhyloInfo {
            msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
            tree,
        };
        let cost = DolloParsimonyCost::new(info);

        assert_eq!(cost.score(), 4.0);
        assert_eq!(cost.score(), 4.0);

        cost.tmp.borrow_mut().node_info_valid.fill(false);
        assert_eq!(cost.score(), 4.0);
    }

    #[test]
    fn dollo_parsimony_simple_scoring() {
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"G-GAA"),
            record!("B", b"G-GG-"),
            record!("C", b"AGCAA"),
            record!("D", b"AGCGA"),
        ]);

        let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

        let info = PhyloInfo {
            msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
            tree,
        };
        let scoring = SimpleScoring::new(1.0, GapCost::new(2.5, 1.0));
        let cost = DolloParsimonyCost::with_scoring(info, scoring.clone());

        assert_eq!(cost.score(), 2.5 + 4.0);
    }

    #[test]
    fn dollo_parsimony_simple_scoring_default() {
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"G-GAA"),
            record!("B", b"G-GG-"),
            record!("C", b"AGCAA"),
            record!("D", b"AGCGA"),
        ]);

        let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

        let info = PhyloInfo {
            msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
            tree,
        };

        let cost = DolloParsimonyCost::new(info.clone());
        assert_eq!(cost.score(), 5.0);
        let scoring = SimpleScoring::new(1.0, GapCost::new(1.0, 1.0));
        let cost2 = DolloParsimonyCost::with_scoring(info, scoring);
        assert_eq!(cost.score(), cost2.score());
    }

    #[test]
    fn tree_upd_testing() {
        let seqs = Sequences::new_unchecked(vec![
            record!("A", b"G--G"),
            record!("B", b"---G"),
            record!("C", b"-GG-"),
            record!("D", b"--GG"),
        ]);

        let tree = tree!("((A:1.0,B:1.0)I1:1.0,(C:1.0,D:1.0)I4:1.0)I0:0.0;");

        let info = PhyloInfo {
            msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
            tree: tree.clone(),
        };
        let mut cost = DolloParsimonyCost::new(info.clone());

        assert_eq!(cost.score(), 1.0);

        cost.update_tree(tree);
        assert_eq!(cost.score(), 1.0);
    }

    #[test]
    fn dollo_multiple_deletions() {
        let seqs = Sequences::with_alphabet_unchecked(
            vec![
                record!("A", b"G"),
                record!("B", b"G"),
                record!("C", b"-"),
                record!("D", b"-"),
            ],
            Alphabet::dna(),
        );

        let tree = tree!("(((D:1,(A:1,C:0.5)I1:0.5)I4:1,B:2)I0:0);");

        let info = PhyloInfo {
            msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
            tree: tree.clone(),
        };

        let scoring = SimpleScoring::new(1.0, GapCost::new(2.0, 1.0));
        let mut cost = DolloParsimonyCost::with_scoring(info, scoring);

        assert_eq!(cost.score(), 4.0);

        cost.update_tree(tree);
        assert_eq!(cost.score(), 4.0);
    }
}
