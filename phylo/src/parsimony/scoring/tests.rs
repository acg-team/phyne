use approx::assert_relative_eq;
use rstest::rstest;

use crate::alphabets::{ParsimonySet, AMINOACIDS, NUCLEOTIDES};
use crate::parsimony::{
    DiagonalZeros as Z, GapCost, ModelScoringBuilder as MCB, ParsimonyScoring, Rounding as R,
    SimpleScoring,
};
use crate::substitution_models::*;

#[test]
fn default_costs() {
    let mismatch = 1.0;
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };

    let c = SimpleScoring::new(mismatch, gap);
    assert_eq!(c.r#match(1.0, &b'A', &b'B'), mismatch);
    assert_eq!(c.r#match(1.0, &b'B', &b'A'), mismatch);
    assert_eq!(c.r#match(1.0, &b'N', &b'K'), mismatch);
    assert_eq!(c.r#match(1.0, &b'C', &b'C'), 0.0);
    assert_eq!(c.avg(2.0), mismatch);
    assert_eq!(c.gap_ext(3.0), gap.ext);
    assert_eq!(c.gap_open(2.0), gap.open);
}

#[test]
fn simple_costs() {
    let mismatch = 3.0;
    let gap = GapCost {
        open: 2.0,
        ext: 10.5,
    };

    let c = SimpleScoring::new(mismatch, gap);
    for char in AMINOACIDS.iter() {
        for char2 in AMINOACIDS.iter() {
            if char == char2 {
                assert_eq!(c.r#match(1.0, char, char2), 0.0);
            } else {
                assert_eq!(c.r#match(1.0, char, char2), mismatch);
            }
        }
    }
    assert_eq!(c.avg(4.9), mismatch);
    assert_eq!(c.gap_open(1.3), gap.open * mismatch);
    assert_eq!(c.gap_ext(2.4), gap.ext * mismatch);

    let mismatch = 1.0;

    let c = SimpleScoring::new(mismatch, gap);
    for char in NUCLEOTIDES.iter() {
        for char2 in NUCLEOTIDES.iter() {
            if char == char2 {
                assert_eq!(c.r#match(1.0, char, char2), 0.0);
            } else {
                assert_eq!(c.r#match(1.0, char, char2), mismatch);
            }
        }
    }
    assert_eq!(c.avg(1.0), mismatch);
    assert_eq!(c.gap_open(1.0), gap.open);
    assert_eq!(c.gap_ext(1.0), gap.ext);
}

#[test]
fn protein_branch_scoring() {
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };

    let times = vec![0.1, 0.7];

    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let cost = MCB::new(model.clone())
        .gap_cost(gap)
        .rounding(R::zero())
        .times(times.clone())
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_07 = 4.0075;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let cost = MCB::new(model)
        .gap_cost(gap)
        .diagonal(Z::zero())
        .rounding(R::zero())
        .times(times)
        .build()
        .unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'A', &b'A'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'A', &b'A'), 0.0);
}

#[test]
fn protein_scoring() {
    let gap = GapCost {
        open: 2.0,
        ext: 0.1,
    };

    let times = vec![0.1, 0.3, 0.5, 0.7];
    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();
    let cost = MCB::new(model)
        .gap_cost(gap)
        .rounding(R::zero())
        .times(times)
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_03 = 4.7475;
    let avg_05 = 4.2825;
    let avg_07 = 4.0075;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);
    assert_eq!(cost.avg(0.3), avg_03);
    assert_eq!(cost.avg(0.5), avg_05);
    assert_eq!(cost.avg(0.7), avg_07);
}

#[test]
fn protein_branch_scoring_nearest() {
    let gap = GapCost {
        open: 2.0,
        ext: 0.1,
    };
    let times = vec![0.1, 0.5];
    let model = SubstModel::<WAG>::new(&[], &[]).unwrap();

    let cost = MCB::new(model)
        .gap_cost(gap)
        .times(times)
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 5.7675;
    let avg_05 = 4.2825;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);
    assert_eq!(cost.avg(0.1), cost.avg(0.2));
    assert_eq!(cost.gap_ext(0.1), cost.gap_ext(0.2));
    assert_eq!(cost.gap_open(0.1), cost.gap_open(0.2));

    assert_eq!(cost.avg(0.005), avg_01);
    assert_eq!(cost.avg(0.15), avg_01);

    assert_eq!(cost.avg(0.45), avg_05);
    assert_eq!(cost.avg(0.5), avg_05);
    assert_eq!(cost.avg(100.0), avg_05);
}

#[test]
fn dna_branch_scoring() {
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };
    let times = vec![0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let cost = MCB::new(model.clone())
        .gap_cost(gap)
        .times(times.clone())
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 2.25;
    let avg_07 = 1.75;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.7), avg_07);

    let cost = MCB::new(model.clone())
        .gap_cost(gap)
        .times(times)
        .diagonal(Z::zero())
        .rounding(R::zero())
        .build()
        .unwrap();

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.r#match(0.1, &b'N', &b'N'), 0.0);

    assert_ne!(cost.avg(0.7), avg_07);
    assert_eq!(cost.r#match(0.7, &b'N', &b'N'), 0.0);
}

#[test]
fn dna_branch_scoring_nearest() {
    let gap = GapCost {
        open: 3.0,
        ext: 0.75,
    };
    let times = vec![0.1, 0.7];

    let model = SubstModel::<JC69>::new(&[], &[]).unwrap();
    let cost = MCB::new(model)
        .gap_cost(gap)
        .times(times)
        .rounding(R::zero())
        .build()
        .unwrap();

    let avg_01 = 2.25;
    let avg_07 = 1.75;

    assert_eq!(cost.avg(0.1), avg_01);
    assert_eq!(cost.gap_ext(0.1), avg_01 * gap.ext);
    assert_eq!(cost.gap_open(0.1), avg_01 * gap.open);

    assert_eq!(cost.avg(0.8), avg_07);

    assert_eq!(cost.avg(0.5), avg_07);
}

#[test]
fn display_gap_costs() {
    let gap = GapCost {
        open: 1.6,
        ext: 1.3,
    };
    let display = format!("{gap}");
    assert!(display.contains("Gap cost multipliers"));
    assert!(display.contains("open: 1.6"));
    assert!(display.contains("ext: 1.3"));
}

#[test]
fn display_simple_scorings() {
    let mismatch = 3.2;
    let gap = GapCost {
        open: 4.6,
        ext: 1.5,
    };
    let display = format!("{}", SimpleScoring::new(mismatch, gap));
    assert!(display.contains("Simple parsimony scoring"));
    assert!(display.contains("mismatch: 3.2"));
    assert!(display.contains("open: 4.6"));
    assert!(display.contains("ext: 1.5"));
}

#[test]
fn display_model_scorings() {
    let model = SubstModel::<GTR>::new(&[], &[]).unwrap();
    let gap = GapCost {
        open: 2.4,
        ext: 1.2,
    };
    let cost = MCB::new(model)
        .gap_cost(gap)
        .times(vec![1.0])
        .build()
        .unwrap();

    let display = format!("{cost}");
    assert!(display.contains("Model-based parsimony scoring"));
    assert!(display.contains("GTR"));
    assert!(display.contains("open: 2.4"));
    assert!(display.contains("ext: 1.2"));
}

#[cfg(test)]
fn min_match_model_template<Q: QMatrix + QMatrixMaker>(set1: &ParsimonySet, set2: &ParsimonySet) {
    let model = SubstModel::<Q>::new(&[], &[]).unwrap();
    let gap = GapCost {
        open: 2.4,
        ext: 1.2,
    };

    for &blen in &[0.1, 1.0, 3.5, 2.4, 50.0, 34.2] {
        let cost = MCB::new(model.clone())
            .gap_cost(gap)
            .times(vec![blen])
            .build()
            .unwrap();

        let score = set1
            .iter()
            .flat_map(|a| set2.iter().map(|b| cost.r#match(blen, a, b)))
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap();

        assert_relative_eq!(score, cost.min_match(blen, set1, set2))
    }
}

#[rstest]
#[case(b"ACT", b"AT")]
#[case(b"TC", b"AG")]
#[case(b"AC", b"TG")]
#[case(b"AGC", b"T")]
fn min_match_model_dna(#[case] set1: &[u8], #[case] set2: &[u8]) {
    let set1 = &ParsimonySet::from_slice(set1);
    let set2 = &ParsimonySet::from_slice(set2);
    min_match_model_template::<JC69>(set1, set2);
    min_match_model_template::<K80>(set1, set2);
    min_match_model_template::<HKY>(set1, set2);
    min_match_model_template::<TN93>(set1, set2);
    min_match_model_template::<GTR>(set1, set2);
}

#[rstest]
#[case(b"ARNDCQEGHILKMFPSTWYV", b"A")]
#[case(b"LKMFPSTWYV", b"ARNDCQEGHI")]
#[case(b"L", b"FP")]
#[case(b"AGC", b"T")]
fn min_match_model_protein(#[case] set1: &[u8], #[case] set2: &[u8]) {
    let set1 = &ParsimonySet::from_slice(set1);
    let set2 = &ParsimonySet::from_slice(set2);
    min_match_model_template::<WAG>(set1, set2);
    min_match_model_template::<HIVB>(set1, set2);
    min_match_model_template::<BLOSUM>(set1, set2);
}

#[rstest]
#[case(b"ACT", b"AT")]
#[case(b"TC", b"AG")]
#[case(b"AC", b"TG")]
#[case(b"AGC", b"T")]
#[case(b"ARNDCQEGHILKMFPSTWYV", b"A")]
#[case(b"LKMFPSTWYV", b"ARNDCQEGHI")]
#[case(b"L", b"FP")]
fn min_match_basic(#[case] set1: &[u8], #[case] set2: &[u8]) {
    let mismatch = 3.5;
    let gap = GapCost {
        open: 2.5,
        ext: 0.5,
    };
    let set1 = &ParsimonySet::from_slice(set1);
    let set2 = &ParsimonySet::from_slice(set2);
    for &blen in &[0.1, 1.0, 3.5, 2.4, 50.0, 34.2] {
        let cost = SimpleScoring::new(mismatch, gap);
        let score = if (set1 & set2).is_empty() {
            mismatch
        } else {
            0.0
        };
        assert_relative_eq!(score, cost.min_match(blen, set1, set2))
    }
}
