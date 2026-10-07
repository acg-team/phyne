use std::array::from_fn;

use itertools::Itertools;
use lazy_static::lazy_static;

use crate::tkf_model::reestimate::{
    EdgeAssignment, EdgeAssignmentPossibilities, QuartetDelOrNot, QuartetDelOrNotPossibilities,
    QuartetEdges, QuartetEvents, N_EDGES_IN_QUARTET,
};
use crate::tkf_model::Event;
use crate::tree::NodeIdx;

/// Given the presence/absence of chars at `t1`, `t2`, `t3`, and `t4`, provides
/// all possible assignments of chars at `v1` and `v2` that are compatible with
/// Dollo's principle. See [`QuartetEdges`].
const DOLLO_ASSIGNMENTS: [&[EdgeAssignment]; 16] = [
    /* 0000 */ &[(false, false)],
    /* 0001 */ &[(true, true), (false, true), (false, false)],
    /* 0010 */ &[(true, true), (false, true), (false, false)],
    /* 0011 */ &[(true, true), (false, true)],
    /* 0100 */ &[(true, true), (true, false), (false, false)],
    /* 0101 */ &[(true, true)],
    /* 0110 */ &[(true, true)],
    /* 0111 */ &[(true, true)],
    /* 1000 */ &[(true, true), (true, false), (false, false)],
    /* 1001 */ &[(true, true)],
    /* 1010 */ &[(true, true)],
    /* 1011 */ &[(true, true)],
    /* 1100 */ &[(true, true), (true, false)],
    /* 1101 */ &[(true, true)],
    /* 1110 */ &[(true, true)],
    /* 1111 */ &[(true, true)],
];

#[inline]
pub(super) fn possible_assignments_of_edge(
    t1_has_char: bool,
    t2_has_char: bool,
    t3_has_char: bool,
    t4_has_char: bool,
) -> EdgeAssignmentPossibilities {
    let idx = (t1_has_char as usize) << 3
        | (t2_has_char as usize) << 2
        | (t3_has_char as usize) << 1
        | (t4_has_char as usize);

    DOLLO_ASSIGNMENTS[idx]
}

/// The number of bits used to encode a single `del_or_not` possibility in the [`DEL_OR_NOT_TABLE`].
const ENCODING_SIZE: usize = 2;
const VARIABLE_CODE: usize = 0b11;
const DELETION_CODE: usize = 0b01;
const NO_DELETION_CODE: usize = 0b00;
const DEL_OR_NOT_TABLE_SIZE: usize = 1 << (ENCODING_SIZE * N_EDGES_IN_QUARTET);

lazy_static! {
/// A constant table that contains precomputed values for 'del_or_not' combinations that can be
/// queried with [`EdgeSeqsReestimator::prev_compatible_del_or_not_table_idx`] and
/// [`EdgeSeqsReestimator::possible_del_or_not_table_idx`].
    static ref DEL_OR_NOT_TABLE: [QuartetDelOrNotPossibilities; DEL_OR_NOT_TABLE_SIZE] =
        possible_del_or_not_table();
}

// TODO: these are not ensured to be compatible with the previous assignments. Would it be
// worth to filter them here?
// See issue #151 https://github.com/acg-team/rust-phylo/issues/151
pub(super) fn prev_compatible_del_or_not(
    current_events: &QuartetEvents,
    q_del_or_not: &QuartetDelOrNot,
) -> &'static QuartetDelOrNotPossibilities {
    let idx = prev_compatible_del_or_not_table_idx(current_events, q_del_or_not);
    &DEL_OR_NOT_TABLE[idx]
}

fn prev_compatible_del_or_not_table_idx(
    current_events: &QuartetEvents,
    q_del_or_not: &QuartetDelOrNot,
) -> usize {
    let mut idx = 0;
    for i in 0..N_EDGES_IN_QUARTET {
        idx <<= ENCODING_SIZE;
        match current_events[i] {
            Event::Nothing => {
                idx |= if q_del_or_not[i] {
                    DELETION_CODE
                } else {
                    NO_DELETION_CODE
                };
            }
            _ => idx |= VARIABLE_CODE,
        }
    }
    idx
}

pub(super) fn possible_del_or_not(
    events: &QuartetEvents,
    is_first_block: bool,
    quartet_edges: &QuartetEdges,
    root: &NodeIdx,
) -> &'static QuartetDelOrNotPossibilities {
    let idx = possible_del_or_not_table_idx(events, is_first_block, quartet_edges, root);
    &DEL_OR_NOT_TABLE[idx]
}

fn possible_del_or_not_table_idx(
    events: &QuartetEvents,
    is_first_block: bool,
    quartet_edges: &QuartetEdges,
    root: &NodeIdx,
) -> usize {
    let mut idx = 0;
    for (i, edge) in quartet_edges.edges().iter().enumerate() {
        idx <<= ENCODING_SIZE;
        let can_be_varied = matches!(events[i], Event::Nothing) // we can't vary if there is an event
                && !is_first_block // we can't vary at the first block, since there is no event that can be passed through
                && edge != root; // we can't vary since deletions cannot happen above the root
        if can_be_varied {
            idx |= VARIABLE_CODE;
        } else {
            idx |= if matches!(events[i], Event::Deletion) {
                DELETION_CODE
            } else {
                NO_DELETION_CODE
            };
        }
    }
    idx
}

fn possible_del_or_not_table() -> [QuartetDelOrNotPossibilities; DEL_OR_NOT_TABLE_SIZE] {
    let mut table = from_fn(|_| Vec::new());
    let choices_per_edge: Vec<Event> = vec![
        Event::Insertion,
        Event::Deletion,
        Event::Homolog,
        Event::Nothing,
    ];
    let quartet_choices = vec![&choices_per_edge; N_EDGES_IN_QUARTET];
    for combination in quartet_choices.into_iter().multi_cartesian_product() {
        let &[e1, e2, e3, e4, e5] = combination.as_slice() else {
            unreachable!();
        };
        let events: [Event; 5] = [*e1, *e2, *e3, *e4, *e5];
        let idx = events_to_idx(&events);
        let possibilities = possible_del_or_not_for_event(&events);
        if !table[idx].is_empty() {
            // The mapping 'calc_possible_del_or_not_table_idx' from all the possibilities to idx is not unique,
            // since the flag 'deletion or not' does not differentiate between Insertion and Homolog events for example.
            // In that case the mapping 'possible_del_or_not_for_event' should also be the same.
            assert!(table[idx] == possibilities);
        }
        table[idx] = possibilities;
    }
    table
}

fn events_to_idx(events: &QuartetEvents) -> usize {
    let mut idx = 0;
    for event in events {
        idx <<= ENCODING_SIZE;
        match event {
            Event::Nothing => idx |= VARIABLE_CODE,
            Event::Deletion => idx |= DELETION_CODE,
            _ => idx |= NO_DELETION_CODE,
        }
    }
    idx
}

fn possible_del_or_not_for_event(events: &QuartetEvents) -> QuartetDelOrNotPossibilities {
    let mut base = [false; N_EDGES_IN_QUARTET];
    let mut position_to_vary = Vec::with_capacity(N_EDGES_IN_QUARTET);

    // collect for each edge whether we have a choice (whether last event was deletion or not) or not
    for i in 0..N_EDGES_IN_QUARTET {
        let can_be_varied = matches!(events[i], Event::Nothing);
        if can_be_varied {
            position_to_vary.push(i);
        } else {
            // determine the fixed del or not
            base[i] = matches!(events[i], Event::Deletion);
        }
    }
    del_or_not_combinations(&position_to_vary, &base)
}

/// Generates all possible combinations of deletion-or-not for the provided choices,
/// while keeping the no_choices fixed.
///
/// # Example
/// ```rust
/// let base = [true, false, false, true, false];
/// let positions_to_vary = vec![1, 3];
/// ```
/// Keeps all the boolean values in `no_choices` fixed except for indices 1 and 3,
/// which are varied over all possible combinations, i.e., the result will be:
/// ```rust
/// let result = [[true, false, false, false, false],
///               [true, false, false, true, false],
///               [true, true, false, false, false],
///               [true, true, false, true, false]];
/// ```
fn del_or_not_combinations(
    positions_to_vary: &[usize],
    base: &QuartetDelOrNot,
) -> QuartetDelOrNotPossibilities {
    let num_combinations = 1 << positions_to_vary.len();
    let mut all_possibilities = Vec::with_capacity(num_combinations);
    for possibility_idx in 0..num_combinations {
        let mut possibility = *base;
        for (j, &edge_index) in positions_to_vary.iter().enumerate() {
            let bit = (possibility_idx >> j) & 1;
            possibility[edge_index] = bit != 0;
        }
        all_possibilities.push(possibility);
    }
    all_possibilities
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use hashbrown::HashSet;
    use rand::{Rng, SeedableRng};
    use rstest::rstest;

    use crate::alignment::AncestralAlignment;
    use crate::alphabets::Alphabet;
    use crate::random::FakeGenerator;
    use crate::tkf_model::{
        tests::setup_test_phylo, EdgeSeqsReestimator, TKF92IndelCostBuilder, TKFModel,
    };

    use super::*;

    #[rstest]
    #[case(0, 0, 0, 0, vec![(0, 0)])]
    #[case(0, 0, 0, 1, vec![(0, 0), (0, 1), (1, 1)])]
    #[case(0, 0, 1, 0, vec![(0, 0), (0, 1), (1, 1)])]
    #[case(0, 0, 1, 1, vec![(0, 1), (1, 1)])]
    #[case(0, 1, 0, 0, vec![(0, 0), (1, 0), (1, 1)])]
    #[case(0, 1, 0, 1, vec![(1, 1)])]
    #[case(0, 1, 1, 0, vec![(1, 1)])]
    #[case(0, 1, 1, 1, vec![(1, 1)])]
    #[case(1, 0, 0, 0, vec![(0, 0), (1, 0), (1, 1)])]
    #[case(1, 0, 0, 1, vec![(1, 1)])]
    #[case(1, 0, 1, 0, vec![(1, 1)])]
    #[case(1, 0, 1, 1, vec![(1, 1)])]
    #[case(1, 1, 0, 0, vec![(1, 1), (1, 0)])]
    #[case(1, 1, 0, 1, vec![(1, 1)])]
    #[case(1, 1, 1, 0, vec![(1, 1)])]
    #[case(1, 1, 1, 1, vec![(1, 1)])]
    fn tkf_possible_assignments_of_edge(
        #[case] t1_has_char: u8,
        #[case] t2_has_char: u8,
        #[case] t3_has_char: u8,
        #[case] t4_has_char: u8,
        #[case] expected: Vec<(u8, u8)>,
    ) {
        // convert expected to bools and sort

        let mut expected = expected
            .into_iter()
            .map(|(a, b)| (a != 0, b != 0))
            .collect::<Vec<(bool, bool)>>();
        expected.sort();

        let mut result = possible_assignments_of_edge(
            t1_has_char != 0,
            t2_has_char != 0,
            t3_has_char != 0,
            t4_has_char != 0,
        )
        .to_vec();
        result.sort();

        assert_eq!(result, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_no_choice() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![];
        let possibilities = del_or_not_combinations(&choices, &no_choices);
        let expected = vec![[true, false, false, true, false]];
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_one() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![0];
        let mut possibilities = del_or_not_combinations(&choices, &no_choices);
        let mut expected = vec![
            [true, false, false, true, false],
            [false, false, false, true, false],
        ];
        possibilities.sort();
        expected.sort();
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![1, 3];
        let mut possibilities = del_or_not_combinations(&choices, &no_choices);
        let mut expected = vec![
            [true, false, false, false, false],
            [true, false, false, true, false],
            [true, true, false, false, false],
            [true, true, false, true, false],
        ];
        possibilities.sort();
        expected.sort();
        assert_eq!(possibilities, expected);
    }

    #[test]
    fn tkf_reestimate_possibilities_for_choices_all() {
        let no_choices = [true, false, false, true, false];
        let choices = vec![0, 1, 2, 3, 4];
        let possibilities = del_or_not_combinations(&choices, &no_choices);
        let expected = (0..32)
            .map(|i| {
                [
                    (i & 0b00001) != 0,
                    (i & 0b00010) != 0,
                    (i & 0b00100) != 0,
                    (i & 0b01000) != 0,
                    (i & 0b10000) != 0,
                ]
            })
            .collect::<Vec<[bool; 5]>>();

        assert_eq!(possibilities, expected);
    }

    /// Returns all possible combinations of `deletion or not` for each edge in the quartet
    /// given the (current) events taken on those edges.
    #[cfg(test)]
    fn possible_del_or_not_for_event_correct(
        reestimator: &EdgeSeqsReestimator<
            impl TKFModel,
            impl AncestralAlignment,
            impl Rng + SeedableRng,
        >,
        events: &QuartetEvents,
        is_first_block: bool,
    ) -> QuartetDelOrNotPossibilities {
        let mut base = [false; N_EDGES_IN_QUARTET];
        let mut position_to_vary = Vec::with_capacity(N_EDGES_IN_QUARTET);

        // collect for each edge whether we have a choice (whether last event was deletion or not) or not
        for (i, edge) in reestimator.quartet_edges.edges().iter().enumerate() {
            let can_be_varied = matches!(events[i], Event::Nothing) // we can't vary if there is an event
                && !is_first_block // we can't vary at the first block, since there is no event that can be passed through
                && edge != &reestimator.cost.phylo.tree.root; // we can't vary since deletions cannot happen above the root
            if can_be_varied {
                position_to_vary.push(i);
            } else {
                // determine the fixed del or not
                base[i] = matches!(events[i], Event::Deletion);
            }
        }
        del_or_not_combinations(&position_to_vary, &base)
    }

    /// Based on the `current_events` and `del_or_not` finds all compatible previous `del_or_not`.
    #[cfg(test)]
    fn prev_compatible_del_or_not_correct(
        current_del_or_not: &QuartetDelOrNot,
        current_events: &QuartetEvents,
    ) -> QuartetDelOrNotPossibilities {
        let mut base = [false; N_EDGES_IN_QUARTET];
        let mut positions_to_vary = Vec::with_capacity(N_EDGES_IN_QUARTET);
        for i in 0..N_EDGES_IN_QUARTET {
            match current_events[i] {
                // we have a gap col, so we have no event here but pass through the previous one
                Event::Nothing => base[i] = current_del_or_not[i],
                // we have an event here, so we can choose any previous `del_or_not`
                _ => positions_to_vary.push(i),
            };
        }
        del_or_not_combinations(&positions_to_vary, &base)
    }

    #[test]
    fn tkf_del_or_not_table_fill_status() {
        let num_non_empty = DEL_OR_NOT_TABLE.iter().filter(|v| !v.is_empty()).count();
        // for each of the edges we have 3 possibilities: either it's a deletion, or not, or we can vary
        let true_size = 3_usize.pow(N_EDGES_IN_QUARTET as u32);
        assert_eq!(num_non_empty, true_size);
    }

    #[cfg(test)]
    fn all_event_combinations() -> impl Iterator<Item = QuartetEvents> {
        fn to_quartet_events(events: &Vec<Event>) -> QuartetEvents {
            let &[e1, e2, e3, e4, e5] = events.as_slice() else {
                unreachable!(); // since N_EDGES_IN_QUARTET is 5
            };
            [e1, e2, e3, e4, e5]
        }
        let choices_per_edge: Vec<Event> = vec![
            Event::Insertion,
            Event::Deletion,
            Event::Homolog,
            Event::Nothing,
        ];
        let all_edges_choices = vec![choices_per_edge; N_EDGES_IN_QUARTET];
        all_edges_choices
            .clone()
            .into_iter()
            .multi_cartesian_product()
            .map(|combination| to_quartet_events(&combination))
    }

    #[cfg(test)]
    fn all_del_or_not_combinations() -> impl Iterator<Item = QuartetDelOrNot> {
        fn to_quartet_del_or_not(del_or_not: &Vec<bool>) -> QuartetDelOrNot {
            let &[d1, d2, d3, d4, d5] = del_or_not.as_slice() else {
                unreachable!(); // since N_EDGES_IN_QUARTET is 5
            };
            [d1, d2, d3, d4, d5]
        }

        let del_or_not = [false, true];
        let del_or_nor_per_edge = [del_or_not; N_EDGES_IN_QUARTET];
        del_or_nor_per_edge
            .into_iter()
            .multi_cartesian_product()
            .map(|combination| to_quartet_del_or_not(&combination))
    }

    #[test]
    fn tkf_possible_del_or_not_table_idx_without_root() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();
        let rng = &mut FakeGenerator::default();
        // reestimator is initiated with default quartet (i.e., none of the edges are the root)
        let reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        let mut seen_idxs = HashSet::new();
        for is_first_block in [true, false] {
            for events in all_event_combinations() {
                let idx = possible_del_or_not_table_idx(
                    &events,
                    is_first_block,
                    &reestimator.quartet_edges,
                    &reestimator.cost.phylo.tree.root,
                );
                seen_idxs.insert(idx);
                let result_from_table = &DEL_OR_NOT_TABLE[idx];
                let result_from_fn = possible_del_or_not(
                    &events,
                    is_first_block,
                    &reestimator.quartet_edges,
                    &reestimator.cost.phylo.tree.root,
                );
                let correct =
                    &possible_del_or_not_for_event_correct(&reestimator, &events, is_first_block);
                assert_eq!(correct, result_from_table);
                assert_eq!(correct, result_from_fn);
            }
        }
        // for each of the edges we have 3 possibilities: either it's a deletion, or not, or we can vary
        let true_size = 3_usize.pow(N_EDGES_IN_QUARTET as u32);
        assert_eq!(seen_idxs.len(), true_size);
    }

    #[test]
    fn tkf_possible_del_or_not_table_idx_with_root() {
        let phylo = setup_test_phylo(Alphabet::dna());
        let root = phylo.tree.root;
        let mut cost = TKF92IndelCostBuilder::new(&[0.4, 0.5, 0.8], phylo)
            .build()
            .unwrap();
        let rng = &mut FakeGenerator::default();
        // reestimator is initiated with default quartet (i.e., none if the edges are root)
        let mut reestimator = EdgeSeqsReestimator::new(&mut cost, rng);
        // preparing for DP initializes the quartet edges such that one of the edges/nodes is the
        // root, thereby affecting the possible idx values
        let node_idx = reestimator.cost.phylo.tree.by_id("I3").idx;
        reestimator.prepare_for_dp(&node_idx);

        let mut seen_idxs = HashSet::new();
        for is_first_block in [true, false] {
            for events in all_event_combinations() {
                let idx = possible_del_or_not_table_idx(
                    &events,
                    is_first_block,
                    &reestimator.quartet_edges,
                    &reestimator.cost.phylo.tree.root,
                );
                seen_idxs.insert(idx);
                let result_from_table = &DEL_OR_NOT_TABLE[idx];
                let result_from_fn =
                    possible_del_or_not(&events, is_first_block, &reestimator.quartet_edges, &root);
                let correct =
                    &possible_del_or_not_for_event_correct(&reestimator, &events, is_first_block);
                assert_eq!(correct, result_from_table);
                assert_eq!(correct, result_from_fn);
            }
        }
        // for each of the edges we have 3 possibilities, except for the root, for which the
        // del_or_not cant be varied, since chars at the root can't be deleted, (although the code
        // for the idx calculation still differentiates between deletion and no deletion even
        // though at the root there can't be a deletion)
        let true_size_for_root = 3_usize.pow((N_EDGES_IN_QUARTET - 1) as u32) * 2;
        assert_eq!(seen_idxs.len(), true_size_for_root);
    }

    #[test]
    fn tkf_prev_compatible_del_or_not_table_idx() {
        let mut seen_idxs = HashSet::new();
        for events in all_event_combinations() {
            for del_or_not in all_del_or_not_combinations() {
                let idx = prev_compatible_del_or_not_table_idx(&events, &del_or_not);
                seen_idxs.insert(idx);
                let result_from_table = &DEL_OR_NOT_TABLE[idx];
                let result_from_fn = prev_compatible_del_or_not(&events, &del_or_not);
                let correct = &prev_compatible_del_or_not_correct(&del_or_not, &events);
                assert_eq!(correct, result_from_table);
                assert_eq!(correct, result_from_fn);
            }
        }
        // for each of the edges we have 3 possibilities: either it's a deletion, or not, or we can vary
        let true_size = 3_usize.pow(N_EDGES_IN_QUARTET as u32);
        assert_eq!(seen_idxs.len(), true_size);
    }
}
