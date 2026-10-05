use crate::alignment::{Alignment, Sequences, MSA};
use crate::likelihood::TreeSearchCost;
use crate::parsimony::{
    scoring::{GapCost, ModelScoringBuilder, SimpleScoring},
    BasicParsimonyCost, DolloParsimonyCost,
};
use crate::phylo_info::PhyloInfo;
use crate::substitution_models::{SubstModel, WAG};
use crate::{record_wo_desc as rec, tree};

#[test]
fn basic_parsimony_cost() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree2 = tree!("((A:1.0,C:1.0):1.0,(B:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree2).unwrap(),
        tree: tree2,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -5.0);

    let tree3 = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree3).unwrap(),
        tree: tree3,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -6.0);
}

#[test]
fn basic_parsimony_cost_tree_update() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree: tree.clone(),
    };
    let mut cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);
    cost.update_tree(tree);
    assert_eq!(cost.cost(), -4.0);
}

#[test]
fn basic_parsimony_reroot() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -4.0);

    let tree_reroot = tree!("(((D:1,(B:1,A:0.5):0.5):1,C:2):0);");

    let info_reroot = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree_reroot).unwrap(),
        tree: tree_reroot,
    };
    let cost_reroot = BasicParsimonyCost::new(info_reroot).unwrap();
    assert_eq!(cost_reroot.cost(), -4.0);
    assert_eq!(cost_reroot.cost(), cost.cost());
}

#[test]
fn basic_parsimony_cost_gaps() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GA-T-"),
        rec!("B", b"GA-TT"),
        rec!("C", b"-A-TT"),
        rec!("D", b"-G-TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let cost = BasicParsimonyCost::new(info).unwrap();
    assert_eq!(cost.cost(), -3.0);
    assert_eq!(cost.cost(), -3.0);
}

#[test]
fn dollo_parsimony_cost_nogaps() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GGA"),
        rec!("B", b"GGG"),
        rec!("C", b"ACA"),
        rec!("D", b"ACG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(1.0, 1.0));
    let cost = DolloParsimonyCost::with_scoring(info, scoring.clone());
    assert_eq!(cost.cost(), -4.0);

    let tree2 = tree!("((A:1.0,C:1.0):1.0,(B:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree2).unwrap(),
        tree: tree2,
    };
    let cost = DolloParsimonyCost::with_scoring(info, scoring.clone());
    assert_eq!(cost.cost(), -5.0);

    let tree3 = tree!("((A:1.0,D:1.0):1.0,(C:1.0,B:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree3).unwrap(),
        tree: tree3,
    };
    let cost = DolloParsimonyCost::with_scoring(info, scoring);
    assert_eq!(cost.cost(), -6.0);
}

#[test]
fn dollo_parsimony_reroot_simple() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GGA-"),
        rec!("B", b"GGG-"),
        rec!("C", b"ACAG"),
        rec!("D", b"ACGG"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(1.0, 1.0));
    let cost = DolloParsimonyCost::with_scoring(info, scoring.clone());
    assert_eq!(cost.cost(), -4.0);

    let tree_reroot = tree!("(((D:1,(B:1,A:0.5):0.5):1,C:2):0);");

    let info_reroot = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree_reroot).unwrap(),
        tree: tree_reroot,
    };
    let cost_reroot = DolloParsimonyCost::with_scoring(info_reroot, scoring);
    // This is a non-reversible cost function.
    assert_eq!(cost_reroot.cost(), -5.0);
    assert_ne!(cost_reroot.cost(), cost.cost());
}

#[test]
fn dollo_parsimony_cost_gaps() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"GA-T-"),
        rec!("B", b"GA-TT"),
        rec!("C", b"-A-TT"),
        rec!("D", b"-A-TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(1.0, 1.0));
    let cost = DolloParsimonyCost::with_scoring(info, scoring);
    assert_eq!(cost.cost(), -1.0);
    assert_eq!(cost.cost(), -1.0);
}

#[test]
fn dollo_parsimony_cost_tree_update() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"G"),
        rec!("B", b"G"),
        rec!("C", b"-"),
        rec!("D", b"-"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree: tree.clone(),
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(2.0, 1.0));
    let mut cost = DolloParsimonyCost::with_scoring(info, scoring);
    assert_eq!(cost.cost(), -0.0);
    cost.update_tree(tree);
    assert_eq!(cost.cost(), 0.0);
}

#[test]
fn dollo_parsimony_cost_deletions() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"T-"),
        rec!("B", b"TT"),
        rec!("C", b"T-"),
        rec!("D", b"TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(2.0, 1.0));
    let cost = DolloParsimonyCost::with_scoring(info, scoring);
    assert_eq!(cost.cost(), -4.0);
}

#[test]
fn dollo_parsimony_cost_low_insertion() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"-"),
        rec!("B", b"T"),
        rec!("C", b"-"),
        rec!("D", b"T"),
        rec!("E", b"-"),
        rec!("F", b"-"),
    ]);
    let tree = tree!("(((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):1.0,(E:1.0,F:1.0):1.0):0.0;");

    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(1.0, 1.0));
    let cost = DolloParsimonyCost::with_scoring(info, scoring);
    assert_eq!(cost.cost(), -2.0);
}

#[test]
fn dollo_parsimony_display() {
    let seqs = Sequences::new_unchecked(vec![
        rec!("A", b"T-"),
        rec!("B", b"TT"),
        rec!("C", b"T-"),
        rec!("D", b"TT"),
    ]);
    let tree = tree!("((A:1.0,B:1.0):1.0,(C:1.0,D:1.0):1.0):0.0;");
    let info = PhyloInfo {
        msa: MSA::from_aligned(seqs.clone(), &tree).unwrap(),
        tree,
    };
    let scoring = SimpleScoring::new(1.0, GapCost::new(2.4, 2.4));
    let cost = DolloParsimonyCost::with_scoring(info.clone(), scoring);
    let display = format!("{cost}");
    assert!(display.contains("Dollo parsimony"));
    assert!(display.contains("Simple parsimony scoring"));
    assert!(display.contains("open: 2.4"));
    assert!(display.contains("ext: 2.4"));

    let model = SubstModel::<WAG>::default();
    let scoring = ModelScoringBuilder::new(model)
        .gap_cost(GapCost::new(1.0, 1.0))
        .times(vec![1.0])
        .build()
        .unwrap();
    let cost = DolloParsimonyCost::with_scoring(info, scoring);

    let display = format!("{cost}");
    assert!(display.contains("Dollo parsimony"));
    assert!(display.contains("Model-based parsimony"));
    assert!(display.contains("WAG"));
    assert!(display.contains("open: 1"));
    assert!(display.contains("ext: 1"));
}
