use std::fs::{self, OpenOptions};
use std::io::{self, BufWriter, Write};
use std::path::Path;

use approx::assert_relative_eq;
use fixedbitset::FixedBitSet;
use itertools::Itertools;

#[cfg(feature = "multithread-brute-force-asr")]
use rayon::prelude::*;

use crate::alignment::{Alignment, AncestralAlignment, MASA};
use crate::likelihood::ModelSearchCost;
use crate::phylo_info::PhyloInfoBuilder;
use crate::random::FakeGenerator;
use crate::tkf_model::reestimate::cache::possible_assignments_of_edge;
use crate::tkf_model::reestimate::mapping_from_node_seq;
use crate::tkf_model::tests::get_mapping_for_any_node;
use crate::tkf_model::{
    EdgeSeqsReestimator, TKF92IndelAddBlocksCostBuilder, TKF92IndelCostBuilder, TKFIndelCost,
    TKFModel,
};
use crate::tree::NodeIdx;

/// Given all possible edge assignment for each block in the alignment returns a specific edge
/// assignment corresponding to the given index.
///
/// # Arguments
///
/// * `idx` - index to decode
/// * `possible_edge_assignments` - a vector of possible edge assignment for each block in the
///   alignment, the outer vector is over blocks, the inner vector is over possible assignments for
///   the edge.
///
/// # Example
///  ```
///  let possibilities= vec![vec![(true, true), (true, false)], vec![(false, false)]];
///  let first = edge_seqs(0, &possibilities); // returns vec![(true, true), (false, false)]
///  let second = edge_seqs(1, &possibilities); // returns vec![(true, false), (false, false)]
///  ```
#[cfg(test)]
fn edge_seqs(
    mut idx: usize,
    possible_edge_assignments: &Vec<Vec<(bool, bool)>>,
) -> Vec<(bool, bool)> {
    let mut tuple = Vec::with_capacity(possible_edge_assignments.len());
    for poss in possible_edge_assignments {
        let choice = idx % poss.len();
        tuple.push((poss[choice].0, poss[choice].1));
        idx /= poss.len();
    }
    tuple
}

#[test]
fn test_edge_seqs() {
    let possible_edge_assignments = vec![
        vec![(true, true), (true, false), (false, true), (false, false)],
        vec![(true, true), (false, false)],
        vec![(true, true), (true, false), (false, true)],
        vec![(true, true), (false, false)],
        vec![(true, true), (true, false), (false, true), (false, false)],
        vec![(true, true)],
        vec![(true, true), (true, false)],
    ];
    let number_of_possibilities: usize = possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product();
    assert_eq!(number_of_possibilities, 384);
    let mut all_possibilities = Vec::with_capacity(number_of_possibilities);
    for i in 0..number_of_possibilities {
        all_possibilities.push(edge_seqs(i, &possible_edge_assignments));
    }
    let unique: Vec<_> = all_possibilities.iter().unique().collect();
    assert_eq!(unique.len(), number_of_possibilities);
    for possibility in possible_edge_assignments
        .into_iter()
        .multi_cartesian_product()
    {
        assert!(all_possibilities.contains(&possibility));
    }
}

#[cfg(test)]
fn edge_seqs_to_mappings(
    edge_seqs: &[(bool, bool)],
    block_lens: &[usize],
    seq_len: usize,
) -> (Vec<Option<usize>>, Vec<Option<usize>>) {
    let mut v2_fixed_bit_set = FixedBitSet::with_capacity(block_lens.len());
    let mut v1_fixed_bit_set = FixedBitSet::with_capacity(block_lens.len());
    for (site, edge_assignment) in edge_seqs.iter().enumerate() {
        if edge_assignment.0 {
            v1_fixed_bit_set.insert(site);
        }
        if edge_assignment.1 {
            v2_fixed_bit_set.insert(site);
        }
    }
    let new_v1_mapping = mapping_from_node_seq(&v1_fixed_bit_set, block_lens, seq_len);
    let new_v2_mapping = mapping_from_node_seq(&v2_fixed_bit_set, block_lens, seq_len);
    (new_v1_mapping, new_v2_mapping)
}

#[cfg(test)]
fn print_progress(i: usize, number_of_possibilities: usize) {
    if (i + 1).is_multiple_of(10000) {
        let percent = i as f64 / number_of_possibilities as f64 * 100.0;
        print!("\r\x1b[2KBrute force progress for current node {percent:.4}%");
        io::stdout().flush().unwrap();
    }
}

#[cfg(test)]
fn clear_progress_line() {
    print!("\r\x1b[2K");
    io::stdout().flush().unwrap();
}

#[cfg(test)]
fn number_of_possibilities(possible_edge_assignments: &[Vec<(bool, bool)>]) -> usize {
    possible_edge_assignments
        .iter()
        .map(|poss| poss.len())
        .product()
}

/// After updating the mappings for `v2` and its parent `v1`, mark the relevant nodes as dirty
/// so that tmp values in `model_info` are recomputed during the next likelihood calculation.
#[cfg(test)]
fn make_nodes_dirty<T: TKFModel>(cost: &mut TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) {
    for child in &cost.phylo.tree.node(v2_idx).children {
        cost.model_info
            .borrow_mut()
            .valid
            .set(usize::from(child), false);
    }
    cost.model_info
        .borrow_mut()
        .valid
        .set(usize::from(cost.phylo.tree.sibling(v2_idx).unwrap()), false);
}

#[cfg(test)]
fn cost_for_edge_seqs<T: TKFModel>(
    v2_idx: &NodeIdx,
    cost: &mut TKFIndelCost<T, MASA>,
    edge_seqs: &[(bool, bool)],
) -> f64 {
    let v1_idx = &cost.phylo.tree.node(v2_idx).parent.unwrap();
    let block_lens = cost.model_info.borrow().block_lengths.clone();
    let seq_len = cost.phylo.msa.len();

    let (new_v1_mapping, new_v2_mapping) = edge_seqs_to_mappings(edge_seqs, &block_lens, seq_len);
    // The expect() are never triggered, unless something is seriously wrong with the algo.
    cost.phylo
        .msa
        .update_ancestral_map(v1_idx, new_v1_mapping)
        .expect("Could not update ancestral map for v1");
    cost.phylo
        .msa
        .update_ancestral_map(v2_idx, new_v2_mapping)
        .expect("Could not update ancestral map for v2");

    make_nodes_dirty(cost, v2_idx);

    cost.logl()
}

#[cfg(test)]
fn get_edge_assignment_possibilities(
    cost: &TKFIndelCost<impl TKFModel, MASA>,
    v2_idx: &NodeIdx,
) -> Vec<Vec<(bool, bool)>> {
    let blocks = &cost.model_info.borrow().blocks;
    let mut possible_edge_assignments = vec![vec![]; blocks.len()];

    let tree = &cost.phylo.tree;
    let msa = &cost.phylo.msa;
    let v1_idx = &tree.node(v2_idx).parent.unwrap();
    let t1_mapping = if tree.parent(v1_idx).is_some() {
        Some(msa.ancestral_map(&cost.phylo.tree.parent(v1_idx).unwrap()))
    } else {
        None
    };

    let sibling = tree.sibling(v2_idx).unwrap();
    let t2_mapping = get_mapping_for_any_node(&cost.phylo.msa, &sibling);
    let t3_mapping = get_mapping_for_any_node(&cost.phylo.msa, &tree.children(v2_idx)[0]);
    let t4_mapping = get_mapping_for_any_node(&cost.phylo.msa, &tree.children(v2_idx)[1]);

    for (block_id, possible_edge_assignment) in possible_edge_assignments.iter_mut().enumerate() {
        let site = blocks[block_id] - 1;
        let t1_is_char = if let Some(t1_map) = &t1_mapping {
            t1_map[site].is_some()
        } else {
            false
        };
        let t2_is_char = t2_mapping[site].is_some();
        let t3_is_char = t3_mapping[site].is_some();
        let t4_is_char = t4_mapping[site].is_some();
        // TODO: perhaps the return type could be changed such that to_vec is not needed here
        // See issue #151 https://github.com/acg-team/rust-phylo/issues/151
        *possible_edge_assignment =
            possible_assignments_of_edge(t1_is_char, t2_is_char, t3_is_char, t4_is_char).to_vec();
    }
    possible_edge_assignments
}

/// The parameter `possible_edge_assignments` contains for each block all possible edge
/// assignments for the two nodes (v1 and v2).
#[cfg(test)]
fn brute_force_max_for_possibilities_single_thread<T: TKFModel>(
    possible_edge_assignments: Vec<Vec<(bool, bool)>>,
    number_of_possibilities: usize,
    cost: &mut TKFIndelCost<T, MASA>,
    v2_idx: &NodeIdx,
) -> f64 {
    let mut max: f64 = f64::MIN;
    for i in 0..number_of_possibilities {
        print_progress(i, number_of_possibilities);
        let possibility = edge_seqs(i, &possible_edge_assignments);
        let current = cost_for_edge_seqs(v2_idx, cost, &possibility);
        max = max.max(current);
    }
    clear_progress_line();
    max
}

// single thread calculation
#[cfg(test)]
#[cfg(not(feature = "multithread-brute-force-asr"))]
fn brute_force_max<T: TKFModel + Send>(mut cost: TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) -> f64 {
    let possible_edge_assignments = get_edge_assignment_possibilities(&cost, v2_idx);
    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    println!(
        "Total number of possibilities to evaluate: {}",
        number_of_possibilities
    );
    brute_force_max_for_possibilities_single_thread(
        possible_edge_assignments,
        number_of_possibilities,
        &mut cost,
        v2_idx,
    )
}

/// The parameter `possible_edge_assignments` contains for each block all possible edge
/// assignments for the two nodes (v1 and v2).
#[cfg(test)]
#[cfg(all(test, feature = "multithread-brute-force-asr"))]
fn brute_force_max_for_possibilities_multi_thread(
    possible_edge_assignments: Vec<Vec<(bool, bool)>>,
    number_of_possibilities: usize,
    cost: &TKFIndelCost<impl TKFModel + Send, MASA>,
    v2_idx: &NodeIdx,
    num_threads: usize,
) -> f64 {
    let chunk_size = number_of_possibilities.div_ceil(num_threads);
    let cost_clones = vec![cost.clone(); num_threads];
    let chunk_maxes: Vec<f64> = (0..number_of_possibilities)
        .into_par_iter()
        .chunks(chunk_size)
        .enumerate()
        .zip(cost_clones)
        .into_par_iter()
        .map(move |((chunk_id, chunk), mut thread_cost)| {
            let mut max: f64 = f64::MIN;
            for i in chunk {
                if chunk_id == 0 {
                    print_progress(i, number_of_possibilities / num_threads);
                }
                let possibility = edge_seqs(i, &possible_edge_assignments);
                let current = cost_for_edge_seqs(v2_idx, &mut thread_cost, &possibility);
                max = max.max(current);
            }
            max
        })
        .collect();
    clear_progress_line();
    chunk_maxes.into_iter().fold(f64::MIN, |a, b| a.max(b))
}

// multi thread  calculation
#[cfg(test)]
#[cfg(feature = "multithread-brute-force-asr")]
fn brute_force_max<T: TKFModel + Send>(mut cost: TKFIndelCost<T, MASA>, v2_idx: &NodeIdx) -> f64 {
    let possible_edge_assignments = get_edge_assignment_possibilities(&cost, v2_idx);
    let number_of_possibilities = number_of_possibilities(&possible_edge_assignments);
    println!(
        "Total number of possibilities to evaluate: {}, for node {}",
        number_of_possibilities,
        cost.phylo.tree.node(v2_idx).id,
    );
    let num_threads = rayon::current_num_threads() / 2;
    // if the number of possibilities is small, do single thread
    if number_of_possibilities < num_threads * 2 {
        brute_force_max_for_possibilities_single_thread(
            possible_edge_assignments,
            number_of_possibilities,
            &mut cost,
            v2_idx,
        )
    } else {
        brute_force_max_for_possibilities_multi_thread(
            possible_edge_assignments,
            number_of_possibilities,
            &cost,
            v2_idx,
            num_threads,
        )
    }
}

#[cfg(test)]
fn delete_existing_brute_force_max_file(file_path: &Path) {
    if Path::new(file_path).exists() {
        fs::remove_file(file_path).expect("Could not delete existing brute force max file");
    }
}

/// Struct to hold calculated brute force max values for each iteration and node. Is exported to a file.
#[cfg(test)]
struct IterationInfo {
    iteration: usize,
    node_id: String,
    logl: f64,
}

#[cfg(test)]
fn load_precalculated_brute_force_maxes(file_path: &Path, iteration_info: &mut Vec<IterationInfo>) {
    let contents = fs::read_to_string(file_path);
    if contents.is_err() {
        println!("Precalculated brute force maxes file not found at {file_path:?}. Not loading any precalculated values.");
        return;
    }
    let rerun_hint = "Consider rerunning with recompute-brute-force-asr feature";
    for line in contents.unwrap().lines() {
        let parts: Vec<&str> = line.trim().split(',').collect();
        if parts.len() != 3 {
            panic!("Invalid line in precalculated brute force maxes file: {line}. It should contain 3 ',' separated columns. {rerun_hint}");
        }
        let iteration = parts[0]
            .parse::<usize>()
            .expect("Could not parse iteration number. {rerun_hint}");
        let node_id = parts[1].to_string();
        let logl = parts[2]
            .parse::<f64>()
            .expect("Could not parse logl value. {rerun_hint}");
        iteration_info.push(IterationInfo {
            iteration,
            node_id,
            logl,
        });
    }
}

#[cfg(test)]
fn append_result(path: &str, iteration: usize, node: String, logl: f64) -> std::io::Result<()> {
    let file = OpenOptions::new().create(true).append(true).open(path)?;
    let mut writer = BufWriter::new(file);
    writeln!(writer, "{iteration},{node},{logl}")?;
    Ok(())
}

#[cfg(test)]
fn calc_or_lookup_brute_force_max(
    iteration: usize,
    iteration_info: &[IterationInfo],
    precalculated_file: &Path,
    tkf_cost: &TKFIndelCost<impl TKFModel + Send, MASA>,
    node: &NodeIdx,
) -> f64 {
    // lookup of pre-calculated value in iteration_info
    if iteration < iteration_info.len() {
        let saved = &iteration_info[iteration];
        let hint = "Consider rerunning with recompute-brute-force-asr feature";
        assert_eq!(
            saved.iteration, iteration,
            "Mismatched iteration number in save file {:?} at iteration {iteration}. {hint}",
            precalculated_file
        );
        assert_eq!(
            saved.node_id,
            tkf_cost.phylo.tree.node(node).id,
            "Mismatched node id in save file {:?} at iteration {iteration}. {hint}",
            precalculated_file
        );
        saved.logl
    }
    // no pre-calculated value available, calculating it and append to file
    else {
        let max = brute_force_max(tkf_cost.clone(), node);
        append_result(
            precalculated_file.to_str().unwrap(),
            iteration,
            tkf_cost.phylo.tree.node(node).id.clone(),
            max,
        )
        .expect("Could not append brute force max to file");
        max
    }
}

#[test]
#[cfg_attr(feature = "ci_coverage", ignore)]
fn tkf92_reestimate_large_tree_for_file_iterative() {
    let dir = Path::new("data/tkf/reestimate/");
    let msa = dir.join("masa.fasta");
    let tree = dir.join("tree.newick");

    let phylo = PhyloInfoBuilder::with_attrs(msa, tree)
        .build_with_ancestors()
        .unwrap();
    phylo.check_dollos_constraint().unwrap();
    let lambda = 1.0;
    let mu = 2.0;
    let r = 0.3;

    // Handle the pre-calculated brute force maxes file
    let precalculated_file = dir.join("precalculated_brute_force_maxes.csv");
    if cfg!(feature = "recompute-brute-force-asr") {
        delete_existing_brute_force_max_file(&precalculated_file);
    }
    let mut iteration_info = Vec::<IterationInfo>::new();
    load_precalculated_brute_force_maxes(&precalculated_file, &mut iteration_info);

    // the cost to be used for repeated reestimation
    let mut reestimator_cost = TKF92IndelCostBuilder::new(&[lambda, mu, r], phylo.clone())
        .build()
        .unwrap();
    // cloning here to leave the cost in a clean state before reestimation
    let mut prev_logl = reestimator_cost.clone().cost();
    let mut rng = FakeGenerator::default();
    let mut reestimator = EdgeSeqsReestimator::new(&mut reestimator_cost, &mut rng);

    // iterating over nodes multiple times
    let repeat = 5;
    let nodes = phylo
        .tree
        .postorder()
        .iter()
        .filter(|node| *node != &phylo.tree.root && !phylo.tree.node(node).children.is_empty())
        .collect::<Vec<_>>()
        .repeat(repeat);

    // Since the inference of the observed blocks may change after reestimation (an so does the
    // cost calculation of a clean cost struct that we use to compare against), we store the
    // initial blocks here and create the clean cost to compare against with these as additional
    // blocks.
    let initial_msa_blocking = reestimator.cost.model_info.borrow().blocks.clone();
    for (iteration, node) in nodes.into_iter().enumerate() {
        // Perform Dynamic Programming reestimation
        let max_dp = reestimator.reestimate_unchecked(node);
        // Perform brute force calculation
        let cost_for_brute_force = TKF92IndelAddBlocksCostBuilder::new(
            &[lambda, mu, r],
            initial_msa_blocking.clone(),
            reestimator.phylo().clone(),
        )
        .build()
        .unwrap();
        let _ = cost_for_brute_force.cost();
        let max_brute_force = calc_or_lookup_brute_force_max(
            iteration,
            &iteration_info,
            &precalculated_file,
            &cost_for_brute_force.clone(),
            node,
        );
        // Create a clean cost to compare against
        let clean_cost = TKF92IndelAddBlocksCostBuilder::new(
            &[lambda, mu, r],
            initial_msa_blocking.clone(),
            reestimator.phylo().clone(),
        )
        .build()
        .unwrap();
        let logl_from_clean_cost = clean_cost.cost();

        assert_relative_eq!(max_dp, logl_from_clean_cost, epsilon = 1e-10);
        assert_relative_eq!(max_dp, max_brute_force, epsilon = 1e-10);
        assert_ne!(max_dp, f64::NEG_INFINITY);
        assert!(logl_from_clean_cost >= prev_logl);
        prev_logl = logl_from_clean_cost;
    }
}
