use std::cell::RefCell;
use std::collections::VecDeque;

use hashbrown::HashMap;
use log::debug;
use rand::{Rng, SeedableRng};
use rand_distr::Geometric;

use crate::alignment::{AlignmentSimulation, AncestralAlignment, Sequences};
use crate::alphabets::{AMB_CHAR, GAP};
use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
use crate::random::RandomGenerator;
use crate::tkf_model::simulate_msa::{Fragmentation, TKFSimulationResult};
use crate::tkf_model::{ln_beta, ln_h1, ln_n0, TKFModel};
use crate::tree::{NodeIdx, Tree};
use crate::{bug, record_wo_desc as record, Result};

/// Abstracts over how a single fragment's length is sampled.
///
/// [TKF92](`crate::tkf_model::TKF92IndelModel`) draws from a geometric distribution parameterised by `r`.
/// [TKF91](`crate::tkf_model::TKF91IndelModel`) always returns length 1 (each residue is its own independent link).
pub trait FragmentSampler {
    /// Samples the length of a single fragment from the model's fragment length
    /// distribution.
    ///
    /// Returns `(length, log_probability)` where `length` is the number of characters in the
    /// fragment (at least 1) and `log_probability` is the natural log of the probability of
    /// drawing that length, so that callers can accumulate a simulation log-likelihood.
    fn sample_fragment_length<R: Rng + SeedableRng>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> (usize, f64);
}

/// Computes the expected length of the root sequence (number of characters)
/// under the TKF model if the process has reached stationarity.
pub trait ExpectedRootLength {
    /// Returns the expected root length under the stationary distribution of the model.
    fn expected_root_length(&self) -> f64;
}

/// Since these sequences are built incrementally, we can't use the [`Sequences`] which hold immutable [records](`record`).
type Seqs = HashMap<NodeIdx, Vec<u8>>;

/// How the length of the root sequence is determined during simulation.
///
/// Used by [`TKFIndelMSASimulator::root_length`] and
/// [`TKFMSASimulator::root_length`](`crate::tkf_model::simulate_msa::TKFMSASimulator::root_length`).
#[derive(Clone, Copy, Debug)]
pub enum RootLength {
    /// Sample the number of root links from the model's geometric distribution.
    Sampled,
    /// Use exactly the given number of characters.
    Defined(usize),
    /// Use the expected root length under the model, rounded to the nearest integer.
    Expected,
}

/// Simulates the indel process under a [TKFModel](crate::tkf_model::TKFModel) to produce an [`AncestralAlignment`]
/// containing only [`GAP`]s and [`AMB_CHAR`]s, representing the indel history.
/// The `max_insertion_length` parameter controls the maximum number of inserted links (=fragments)
/// that can be produced in a single insertion event.
/// Note that, the alignment might contain columns where the character goes extinct, i.e., all leaf
/// sequences have a gap in that column. You may want to call
/// [`TKFSimulationResult::remove_extinct_columns`](`TKFSimulationResult::remove_extinct_columns`).
///
/// # Example
/// ```
/// use phyne::alignment::{Alignment, AncestralAlignment, MASA};
/// use phyne::random::DefaultGenerator;
/// use phyne::tkf_model::simulate_msa::{RootLength, TKFIndelMSASimulator};
/// use phyne::tkf_model::TKF92IndelModel;
/// use phyne::tree;
/// # use phyne::Result;
///
/// # fn main() -> Result<()> {
/// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
/// // The maximum number of inserted links (=fragments) in a single insertion event.
/// let max_ins_len = Some(50);
/// let mut simulator = TKFIndelMSASimulator::new(
///     TKF92IndelModel::default(),
///     tree,
///     max_ins_len,
/// )?;
/// // Fix the root sequence to exactly 100 characters.
/// simulator.root_length(RootLength::Defined(100));
/// let seed = 123;
/// let mut rng = DefaultGenerator::new(seed);
/// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
/// assert_eq!(result.phylo.msa.seq_count(), 3);
/// let root_map = result.phylo.msa.ancestral_map(&result.phylo.tree.root);
/// assert_eq!(root_map.iter().filter(|site| site.is_some()).count(), 100);
/// # Ok(()) }
/// ```
pub struct TKFIndelMSASimulator<T: TKFModel + FragmentSampler + ExpectedRootLength> {
    indel_model: T,
    tree: Tree,
    cumulative_logl: RefCell<f64>,
    // The maximum number of inserted links (=fragments) in a single insertion event.
    // If `None`, then there is no limit on the number of inserted links.
    max_inserted_frag_len: usize,
    root_length: RootLength,
}

impl<T, R> AlignmentSimulation<R> for TKFIndelMSASimulator<T>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
    R: Rng + SeedableRng,
{
    fn simulate_ancestral_alignment<AA: AncestralAlignment>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> PhyloInfo<AA> {
        let TKFSimulationResult { phylo, .. } = self.simulate_with_fragments(rng);
        phylo
    }

    fn tree(&self) -> &Tree {
        &self.tree
    }
}

/// All the descendant links associated to a single branch.
type BranchLinkChildren = Vec<TKFLink>;

struct TKFLink {
    /// The node in the tree the link is associated with.
    node: NodeIdx,
    /// True if the link is immortal. It's the link to the very left of the sequence.
    is_immortal: bool,
    /// How many characters are associated to the right of this link.
    length: usize,
    /// For every child node/branch of this link's node, the descendant links.
    children: Vec<BranchLinkChildren>,
    /// For every child node/branch of this link's node, the [`LinkFate`] of this link.
    fates: Vec<LinkFate>,
    /// Whether this link was produced by an insertion event.
    is_insertion: bool,
}

impl TKFLink {
    fn new_immortal(node: NodeIdx) -> Self {
        TKFLink {
            node,
            is_immortal: true,
            length: 0,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }

    fn new(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: false,
        }
    }

    fn new_insertion(node: NodeIdx, length: usize) -> Self {
        TKFLink {
            node,
            is_immortal: false,
            length,
            children: Vec::new(),
            fates: Vec::new(),
            is_insertion: true,
        }
    }
}

#[derive(Clone)]
enum LinkFate {
    /// The link is deleted.
    Deletion,
    /// The link survives on the branch and produces `usize` many insertions to the right of it.
    Homolog(usize),
    /// The link is deleted but produces `usize` many insertions to the right of it.
    NonHomolog(usize),
}

impl<T> TKFIndelMSASimulator<T>
where
    T: TKFModel + FragmentSampler + ExpectedRootLength,
{
    /// Creates a new [`TKFIndelMSASimulator`] with the given indel model, tree and the
    /// maximum number of inserted links (=fragments) in a single insertion event.
    ///
    /// # Example
    /// ```
    /// use phyne::tkf_model::simulate_msa::TKFIndelMSASimulator;
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let simulator = TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
    /// assert_eq!(simulator.tree().len(), 5);
    /// # Ok(()) }
    /// ```
    ///
    /// # Errors
    /// * If the tree node IDs cannot be completed (see
    ///   [`set_missing_tree_node_ids`](`crate::phylo_info::set_missing_tree_node_ids`)).
    pub fn new(
        indel_model: T,
        mut tree: Tree,
        max_insertion_length: Option<usize>,
    ) -> Result<Self> {
        set_missing_tree_node_ids(&mut tree)?;
        Ok(Self {
            indel_model,
            tree,
            cumulative_logl: RefCell::new(0.0),
            // If `None`, then there is no limit. Since we cast to `i32` in the sampling functions,
            // we use `i32::MAX` as the limit.
            max_inserted_frag_len: max_insertion_length.unwrap_or(i32::MAX as usize),
            root_length: RootLength::Sampled,
        })
    }

    /// Sets the constraint for the [`RootLength`] used for the simulation.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{AncestralAlignment, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::tkf_model::simulate_msa::{RootLength, TKFIndelMSASimulator};
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let mut simulator =
    ///     TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
    /// let root_len = 100;
    /// simulator.root_length(RootLength::Defined(root_len));
    ///
    /// let mut rng = DefaultGenerator::new(123);
    /// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    /// let root_map = result.phylo.msa.ancestral_map(&result.phylo.tree.root);
    /// assert_eq!(root_map.iter().filter(|site| site.is_some()).count(), root_len);
    /// # Ok(()) }
    /// ```
    pub fn root_length(&mut self, root_length: RootLength) {
        self.root_length = root_length;
    }

    /// Returns the tree the simulation is performed on.
    ///
    /// # Example
    /// ```
    /// use phyne::tkf_model::simulate_msa::TKFIndelMSASimulator;
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let simulator =
    ///     TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
    /// assert_eq!(simulator.tree().node_id(&simulator.tree().root), "R");
    /// # Ok(()) }
    /// ```
    pub fn tree(&self) -> &Tree {
        &self.tree
    }

    /// Simulates the indel process and returns the alignment, fragmentation, and log-likelihood of the simulated MSA
    /// under the model, which is accumulated during the simulation.
    fn simulate_with_fragments_and_logl<AA: AncestralAlignment, R: Rng + SeedableRng>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> (TKFSimulationResult<AA>, f64) {
        *self.cumulative_logl.borrow_mut() = 0.0;
        let links = self.build_msa_links(rng);
        let (phylo, fragmentation) = self.links_to_msa(&links);
        let result = TKFSimulationResult {
            phylo,
            fragmentation,
        };
        let logl = *self.cumulative_logl.borrow();
        (result, logl)
    }

    /// Simulates the indel process and produces the corresponding ancestral alignment and fragmentation.
    ///
    /// # Example
    /// ```
    /// use phyne::alignment::{Alignment, MASA};
    /// use phyne::random::DefaultGenerator;
    /// use phyne::tkf_model::simulate_msa::TKFIndelMSASimulator;
    /// use phyne::tkf_model::TKF92IndelModel;
    /// use phyne::tree;
    /// # use phyne::Result;
    ///
    /// # fn main() -> Result<()> {
    /// let tree = tree!("((A:1.0,B:1.0)I:1.0,C:1.0)R;");
    /// // The maximum number of inserted links (=fragments) in a single insertion event.
    /// let max_ins_len = Some(50);
    /// let simulator =
    ///     TKFIndelMSASimulator::new(TKF92IndelModel::default(), tree, max_ins_len)?;
    /// let mut rng = DefaultGenerator::new(123);
    /// let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
    ///
    /// assert_eq!(result.phylo.msa.seq_count(), 3);
    /// assert!(!result.fragmentation.is_empty());
    /// # Ok(()) }
    /// ```
    pub fn simulate_with_fragments<AA: AncestralAlignment, R: Rng + SeedableRng>(
        &self,
        rng: &mut RandomGenerator<R>,
    ) -> TKFSimulationResult<AA> {
        self.simulate_with_fragments_and_logl(rng).0
    }

    /// Samples the number of root links from a geometric distribution. Can be zero.
    fn sample_num_root_links<R: Rng + SeedableRng>(&self, rng: &mut RandomGenerator<R>) -> usize {
        let prob_of_success = 1.0 - self.indel_model.lambda() / self.indel_model.mu(); // ie stopping the links
        let geom = Geometric::new(prob_of_success).unwrap();
        let choice = rng.sample(&geom);
        let log_prob = (choice as f64) * (1.0 - prob_of_success).ln() + prob_of_success.ln();
        *self.cumulative_logl.borrow_mut() += log_prob;
        choice as usize
    }

    /// Samples the length of a fragment from the indel model's fragment length distribution, and
    /// accumulates the log-probability of that length into the cumulative log-likelihood of the simulation.
    fn sample_fragment_length<R: Rng + SeedableRng>(&self, rng: &mut RandomGenerator<R>) -> usize {
        let (length, log_prob) = self.indel_model.sample_fragment_length(rng);
        *self.cumulative_logl.borrow_mut() += log_prob;
        length
    }

    /// Builds the links associated with the root of the tree. This includes one immortal
    /// link and a number of mortal links either sampled or defined.
    fn build_root_links<R: Rng + SeedableRng>(&self, rng: &mut RandomGenerator<R>) -> Vec<TKFLink> {
        let min_needed_links = match self.root_length {
            RootLength::Sampled => self.sample_num_root_links(rng),
            RootLength::Defined(len) => len,
            RootLength::Expected => self.indel_model.expected_root_length().round() as usize,
        };
        let max_root_seq_len = match self.root_length {
            RootLength::Defined(len) => Some(len),
            RootLength::Expected => Some(self.indel_model.expected_root_length().round() as usize),
            RootLength::Sampled => None,
        };
        let mut root_links = Vec::with_capacity(min_needed_links + 1); // +1 for the immortal link
        root_links.push(TKFLink::new_immortal(self.tree.root));

        let mut current_total_root_len = 0;
        for _ in 0..min_needed_links {
            let frag_length = self.sample_fragment_length(rng);
            match max_root_seq_len {
                Some(max_len) => {
                    if current_total_root_len + frag_length <= max_len {
                        root_links.push(TKFLink::new(self.tree.root, frag_length));
                        current_total_root_len += frag_length;
                    } else {
                        let remaining_len = max_len - current_total_root_len;
                        // `remaining_len` can be 0 when the fragments exactly fill the root
                        // sequence. A zero-length link would add no characters but would still
                        // push a fragment boundary equal to the previous one, so it is skipped.
                        if remaining_len > 0 {
                            root_links.push(TKFLink::new(self.tree.root, remaining_len));
                        }
                        break;
                    }
                }
                _ => {
                    root_links.push(TKFLink::new(self.tree.root, frag_length));
                }
            }
        }
        root_links
    }

    /// First [builds the root links](`Self::build_root_links`)) and then
    /// [evolves them down the tree](`Self::evolve_link_down_tree`) to produce the full link
    /// structure of the MSA.
    fn build_msa_links<R: Rng + SeedableRng>(&self, rng: &mut RandomGenerator<R>) -> Vec<TKFLink> {
        let mut root_links = self.build_root_links(rng);
        for link in root_links.iter_mut() {
            // TODO: this can be parallelized. Also, the conversion to to alignment, i.e., the
            // progressive appending to it can be done after a root link is done evolving down the
            // tree, which would allow for more efficient memory usage since we won't have to keep
            // the whole link structure in memory at once.
            // See issue #163 https://github.com/acg-team/rust-phylo/issues/163
            self.evolve_link_down_tree(link, rng);
        }
        root_links
    }

    /// Evolves a single link down the tree, by sampling its fate on every branch and creating the
    /// descendant links accordingly.
    fn evolve_link_down_tree<R: Rng + SeedableRng>(
        &self,
        link: &mut TKFLink,
        rng: &mut RandomGenerator<R>,
    ) {
        for (branch_id, child_node) in self.tree.children(&link.node).iter().enumerate() {
            link.children.push(Vec::new());
            let branch_length = self.tree.node(child_node).blen;
            if link.is_immortal {
                //  immortal link always survives and evolves down the tree
                let child_link = TKFLink::new_immortal(*child_node);
                link.children[branch_id].push(child_link);
                self.evolve_link_down_tree(link.children[branch_id].last_mut().unwrap(), rng);
                // new insertions
                let num_insertions = self.sample_tkf_immortal_link_fate(branch_length, rng);
                self.evolve_insertions(num_insertions, link, branch_id, rng);
            } else {
                let fate = self.sample_tkf_link_fate(branch_length, rng);
                link.fates.push(fate.clone());
                match fate {
                    LinkFate::Deletion => {
                        // link is deleted, do nothing
                    }
                    LinkFate::Homolog(num_children) => {
                        // the surviving homologous link
                        let child_link = TKFLink::new(*child_node, link.length);
                        link.children[branch_id].push(child_link);
                        self.evolve_link_down_tree(
                            link.children[branch_id].last_mut().unwrap(),
                            rng,
                        );
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id, rng);
                    }
                    LinkFate::NonHomolog(num_children) => {
                        // original link is deleted, do not evolve_link_down_tree
                        // new insertions
                        self.evolve_insertions(num_children, link, branch_id, rng);
                    }
                }
            }
        }
    }

    /// A helper for [evolve_link_down_tree](`Self::evolve_link_down_tree`) that evolves a `number`
    /// of insertions on a branch, by creating the insertion links and evolving them down the tree
    /// as well.
    fn evolve_insertions<R: Rng + SeedableRng>(
        &self,
        number: usize,
        parent_link: &mut TKFLink,
        branch_id: usize,
        rng: &mut RandomGenerator<R>,
    ) {
        for _ in 0..number {
            let length = self.sample_fragment_length(rng);
            let node = self.tree.children(&parent_link.node)[branch_id];
            let insertion_link = TKFLink::new_insertion(node, length);
            parent_link.children[branch_id].push(insertion_link);
            self.evolve_link_down_tree(parent_link.children[branch_id].last_mut().unwrap(), rng);
        }
    }

    /// Samples the fate of a mortal link on a branch, which includes whether it survives or is
    /// deleted, and how many insertions it produces in either case.
    fn sample_tkf_link_fate<R: Rng + SeedableRng>(
        &self,
        time: f64,
        rng: &mut RandomGenerator<R>,
    ) -> LinkFate {
        let uniform_sample = rng.random::<f64>();
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = ln_beta(lambda, mu, time).exp();
        let n_0 = ln_n0(self.indel_model.mu(), beta.ln()).exp();
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        debug_assert!(
            (n_0 + homolog_prob_integrated + non_homolog_prob_integrated - 1.0).abs() < 1e-10
        );
        // Link is deleted without producing any insertions
        if uniform_sample < n_0 {
            *self.cumulative_logl.borrow_mut() += (n_0).ln();
            LinkFate::Deletion
        } else if uniform_sample < n_0 + homolog_prob_integrated {
            // Link survives homologously
            let adjusted_sample = uniform_sample - n_0;
            self.sample_homolog_fate(time, adjusted_sample)
        } else {
            // Link is deleted but has non-homologous insertions
            debug_assert!(1.0 - uniform_sample < non_homolog_prob_integrated);
            let adjusted_sample = uniform_sample - n_0 - homolog_prob_integrated;
            self.sample_non_homolog_fate(time, adjusted_sample)
        }
    }

    /// Samples the number of homologous insertions on a branch, given that the original link survives.
    fn sample_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = ln_beta(lambda, mu, time).exp();
        let mut cumulative_prob = 0.0;
        for n in 0..self.max_inserted_frag_len {
            // n + 1 because we are counting the original link
            let homolog_prob = homolog_prob(n + 1, lambda, mu, beta, time);
            cumulative_prob += homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
                return LinkFate::Homolog(n);
            }
        }
        // didn't return in the loop, capping insertion at length max_insertion_length
        let homolog_prob_integrated = homolog_prob_integrated(mu, time);
        let homolog_prob = homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += homolog_prob.ln();
        debug!(
            "Capping homologous insertion length at {}",
            self.max_inserted_frag_len
        );
        LinkFate::Homolog(self.max_inserted_frag_len)
    }

    /// Samples the number of non-homologous insertions on a branch, given that the original link is deleted.
    fn sample_non_homolog_fate(&self, time: f64, uniform_sample: f64) -> LinkFate {
        let lambda = self.indel_model.lambda();
        let mu = self.indel_model.mu();
        let beta = ln_beta(lambda, mu, time).exp();
        let mut cumulative_prob = 0.0;
        for n in 1..self.max_inserted_frag_len {
            let non_homolog_prob = non_homolog_prob(n, lambda, mu, beta, time);
            cumulative_prob += non_homolog_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
                return LinkFate::NonHomolog(n);
            }
        }
        // didn't return in the loop, capping insertion at length max_insertion_length
        let non_homolog_prob_integrated = non_homolog_prob_integrated(mu, beta, time);
        let non_homolog_prob = non_homolog_prob_integrated - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += non_homolog_prob.ln();
        debug!(
            "Capping non-homologous insertion length at {}",
            self.max_inserted_frag_len
        );
        LinkFate::NonHomolog(self.max_inserted_frag_len)
    }

    /// Since immortal links always survive, we only need to sample how many insertions they
    /// produce, which is what this function does.
    fn sample_tkf_immortal_link_fate<R: Rng + SeedableRng>(
        &self,
        time: f64,
        rng: &mut RandomGenerator<R>,
    ) -> usize {
        let uniform_sample = rng.random::<f64>();
        let lambda = self.indel_model.lambda();
        let beta = ln_beta(lambda, self.indel_model.mu(), time).exp();
        let mut cumulative_prob = 0.0;
        for n in 0..self.max_inserted_frag_len {
            // n + 1 because we are counting the original immortal link
            let immortal_prob = immortal_prob(n + 1, lambda, beta);
            cumulative_prob += immortal_prob;
            if uniform_sample < cumulative_prob {
                *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
                return n;
            }
        }
        let immortal_prob = 1.0 - cumulative_prob;
        *self.cumulative_logl.borrow_mut() += immortal_prob.ln();
        self.max_inserted_frag_len
    }

    /// After [simulation of the links](`Self::build_msa_links`) is complete, this function
    /// calls [`Self::append_link_to_msa`] for every root link to build the final MSA and fragmentation.
    fn links_to_msa<AA: AncestralAlignment>(
        &self,
        links: &[TKFLink],
    ) -> (PhyloInfo<AA>, Fragmentation) {
        let mut msa: Seqs = HashMap::new();
        for node in self.tree.preorder() {
            msa.insert(*node, Vec::new());
        }
        let mut fragmentation: Vec<usize> = Vec::new();
        for link in links {
            self.append_link_to_msa(link, &mut msa, &mut fragmentation);
        }

        let records = msa
            .iter()
            .map(|(node, seq)| record!(&self.tree.node(node).id, seq))
            .collect();

        let seqs = Sequences::new(records).unwrap_or_else(|e| {
            bug!(
                "Sequences should be valid since they were built from the links",
                e
            )
        });
        let phylo = AA::from_aligned_with_ancestral(seqs, self.tree.clone()).unwrap_or_else(|e| {
            bug!(
                "AncestralAlignment should be valid since it was built from the links",
                e
            )
        });

        let fragmentation = Fragmentation::new(fragmentation);
        fragmentation.fragmentation_works_with_ancestral_alignment(&phylo.msa);
        (phylo, fragmentation)
    }

    /// Traverses the link structure on the tree for every root link and progressively builds the alignment.
    fn append_link_to_msa(&self, link: &TKFLink, msa: &mut Seqs, fragmentation: &mut Vec<usize>) {
        // Insertions are pushed to the front of the queue so that they the lower insertions on the
        // tree are processed before the higher ones, which ensures that gaps are correctly
        // inserted since lower insertions don't mess with the events higher up in the tree.
        // Which is especially important for the case of non-homologous insertions and the use of eta.
        let mut insertions = VecDeque::from([link]);
        while let Some(insertion_link) = insertions.pop_front() {
            // For every insertion event do a BFS traversal, preorder would not work since we need
            // to maintain the priorities of the insertion events, see the comment above.
            let mut tree_stack = VecDeque::from([insertion_link]);
            while let Some(current_link) = tree_stack.pop_front() {
                if current_link.is_immortal {
                    self.dispatch_immortal_children(current_link, &mut tree_stack, &mut insertions);
                } else {
                    self.process_link(
                        current_link,
                        msa,
                        fragmentation,
                        &mut tree_stack,
                        &mut insertions,
                    );
                }
            }
        }
    }

    /// Pushes the children of an immortal link onto the traversal stacks.
    ///
    /// The first child of each branch continues the homolog tree traversal (`tree_stack`) which is
    /// the immortal surviving; any additional children are new insertions and go onto the `insertions` queue.
    fn dispatch_immortal_children<'a>(
        &self,
        link: &'a TKFLink,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(link.is_immortal);
        for branch_child in &link.children {
            for (child_id, child_link) in branch_child.iter().enumerate() {
                if child_id == 0 {
                    tree_stack.push_back(child_link);
                } else {
                    insertions.push_front(child_link);
                }
            }
        }
    }

    /// Writes one non-immortal link's contribution to the MSA and updates the traversal stacks.
    ///
    /// - Appends `AMB_CHAR * length` to the owning node.
    /// - Records a fragmentation boundary for root-owned links and insertion links.
    /// - Fills `NOTHING_CHAR` gaps into all non-owning nodes for insertion links.
    /// - Applies each branch fate: queues homolog/insertion children, or fills deletion gaps.
    fn process_link<'a>(
        &self,
        link: &'a TKFLink,
        msa: &mut Seqs,
        fragmentation: &mut Vec<usize>,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(!link.is_immortal);
        let seq = msa.get_mut(&link.node).unwrap();
        let new_len = seq.len() + link.length;
        seq.resize(new_len, AMB_CHAR);
        if link.node == self.tree.root || link.is_insertion {
            let fragment_boundary = link.length + fragmentation.last().unwrap_or(&0);
            fragmentation.push(fragment_boundary);
        }
        if link.is_insertion {
            self.insertion_gaps(link.length, msa, &link.node);
        }
        for (branch_id, child_links) in link.children.iter().enumerate() {
            self.apply_fate(link, branch_id, child_links, msa, tree_stack, insertions);
        }
    }

    /// Applies a single branch fate for a non-immortal link.
    ///
    /// - `Homolog`: queues the surviving child onto `tree_stack` and its insertions onto `insertions`.
    /// - `Deletion` / `NonHomolog`: fills deletion gaps for all descendants; `NonHomolog` also
    ///   queues the new insertion children.
    fn apply_fate<'a>(
        &self,
        link: &'a TKFLink,
        branch_id: usize,
        child_links: &'a [TKFLink],
        msa: &mut Seqs,
        tree_stack: &mut VecDeque<&'a TKFLink>,
        insertions: &mut VecDeque<&'a TKFLink>,
    ) {
        debug_assert!(!link.is_immortal);
        match &link.fates[branch_id] {
            LinkFate::Homolog(_) => {
                tree_stack.push_back(&child_links[0]);
                for l in child_links.iter().skip(1) {
                    insertions.push_front(l);
                }
            }
            LinkFate::Deletion => {
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
                }
            }
            LinkFate::NonHomolog(_) => {
                for l in child_links {
                    insertions.push_front(l);
                }
                let child_node = self.tree.children(&link.node)[branch_id];
                for descendant_node in self.tree.preorder_subroot(&child_node) {
                    let seq = msa.get_mut(&descendant_node).unwrap();
                    let new_len = seq.len() + link.length;
                    seq.resize(new_len, GAP);
                }
            }
        }
    }

    /// Appends gaps to the progressively build alignment for every node in the tree that is not in
    /// the subtree of the insertion node.
    fn insertion_gaps(&self, length: usize, msa: &mut Seqs, insertion_node: &NodeIdx) {
        self.insertion_gaps_subtree(length, msa, &self.tree.root, insertion_node);
    }

    /// A helper for [insertion_gaps](`Self::insertion_gaps`) that recursively traverses the tree
    /// and appends gaps to the appropriate nodes.
    fn insertion_gaps_subtree(
        &self,
        length: usize,
        msa: &mut Seqs,
        subtree_node: &NodeIdx,
        insertion_node: &NodeIdx,
    ) {
        if subtree_node == insertion_node {
            return;
        }
        let seq = msa.get_mut(subtree_node).unwrap();
        let new_len = seq.len() + length;
        seq.resize(new_len, GAP);
        for child_node in self.tree.children(subtree_node) {
            self.insertion_gaps_subtree(length, msa, child_node, insertion_node);
        }
    }
}

/// Returns the sum [`homolog_prob`] over all `n` (number of insertions) which is the probability
/// that a link survives on a branch of length `time` regardless of how many insertions it produces.   
fn homolog_prob_integrated(mu: f64, time: f64) -> f64 {
    (-mu * time).exp()
}

/// Returns the sum of [`non_homolog_prob`] over all `n` (number of insertions) which is the
/// probability that a link is deleted on a branch of length `time` but produces at least one insertion.
fn non_homolog_prob_integrated(mu: f64, beta: f64, time: f64) -> f64 {
    1.0 - (-mu * time).exp() - mu * beta
}

/// In the TKF model `n` is at least 1 (the original link)
fn homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let h1_val = ln_h1(lambda, mu, beta.ln(), time).exp();
    h1_val * (lambda * beta).powi((n - 1) as i32)
}

/// In the TKF model `n` is at least 1 because otherwise we should have used
/// [`ln_n0`](`crate::tkf_model::ln_n0`).exp().
fn non_homolog_prob(n: usize, lambda: f64, mu: f64, beta: f64, time: f64) -> f64 {
    let t1 = 1.0 - (-mu * time).exp() - mu * beta;
    let t2 = 1.0 - lambda * beta;
    let t3 = (lambda * beta).powi((n - 1) as i32);
    t1 * t2 * t3
}

/// In the TKF model `n` is at least 1 because the immortal link cannot die
fn immortal_prob(n: usize, lambda: f64, beta: f64) -> f64 {
    (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32)
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod private_tests {
    use approx::assert_relative_eq;
    use rstest::rstest;

    use crate::alignment::{Alignment, AncestralAlignment, MASA};
    use crate::phylo_info::{set_missing_tree_node_ids, PhyloInfo};
    use crate::random::DefaultGenerator;
    use crate::tkf_model::{
        TKF91IndelCostBuilder, TKF91IndelModel, TKF92FixedIndelCostBuilder, TKF92IndelModel,
    };
    use crate::tree;

    use super::*;

    #[rstest]
    #[case(0.1, 0.2, 0.5)]
    #[case(0.5, 0.7, 1.0)]
    #[case(1.0, 1.5, 0.0001)]
    fn tkf_integrated_probs(#[case] lambda: f64, #[case] mu: f64, #[case] time: f64) {
        // arrange
        let beta = ln_beta(lambda, mu, time).exp();
        //act
        let homolog_integrated = homolog_prob_integrated(mu, time);
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);
        let n0 = ln_n0(mu, beta.ln()).exp();
        // assert
        assert_relative_eq!(
            homolog_integrated + non_homolog_integrated + n0,
            1.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn tkf_homolog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = ln_beta(lambda, mu, time).exp();
        let n = 4;

        let prob = homolog_prob(n, lambda, mu, beta, time);
        let true_prob =
            (-mu * time).exp() * (1.0 - lambda * beta) * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_non_homolog_probs() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.23;
        let beta = ln_beta(lambda, mu, time).exp();
        let n = 4;

        let prob = non_homolog_prob(n, lambda, mu, beta, time);
        let true_prob = (1.0 - (-mu * time).exp() - mu * beta)
            * (1.0 - lambda * beta)
            * (lambda * beta).powi((n - 1) as i32);
        assert_relative_eq!(prob, true_prob);
    }

    #[test]
    fn tkf_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = ln_beta(lambda, mu, time).exp();
        let homolog_integrated = homolog_prob_integrated(mu, time);

        let mut homolog_sum = 0.0;
        for n in 1..100 {
            homolog_sum += homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(homolog_sum, homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf_non_homolog_probs_sum_to_integrated() {
        let lambda = 0.5;
        let mu = 0.7;
        let time = 1.0;
        let beta = ln_beta(lambda, mu, time).exp();
        let non_homolog_integrated = non_homolog_prob_integrated(mu, beta, time);

        let mut non_homolog_sum = 0.0;
        for n in 1..100 {
            non_homolog_sum += non_homolog_prob(n, lambda, mu, beta, time);
        }

        assert_relative_eq!(non_homolog_sum, non_homolog_integrated, epsilon = 1e-10);
    }

    #[test]
    fn tkf92_indel_simulate() {
        for seed in 0..100 {
            let lambda = 1.1;
            let mu = 1.2;
            let r = 0.6;
            let tkf_model = TKF92IndelModel::new(lambda, mu, r);
            let mut tree = tree!("(((9: 0.220067, (10: 0.150408, 6: 0.150408): 0.069660): 0.054476, (5: 0.033214, 8: 0.033214): 0.241329): 0.725457, ((((3: 0.033362, 1: 0.033362): 0.000893, 7: 0.034255): 0.379610, 4: 0.413865): 0.310280, 2: 0.724145): 0.275855);");
            set_missing_tree_node_ids(&mut tree).unwrap();

            // Should be so large that it doesn't kick in. Because otherwise the
            // accumulated logl would not match the clean cost. Since event probabilities
            // are summed together for the case of drawing a max_insertion_length.
            let max_insertion_length = 100;
            let simulator =
                TKFIndelMSASimulator::new(tkf_model, tree.clone(), Some(max_insertion_length))
                    .unwrap();
            let mut rng = DefaultGenerator::new(seed);
            let (result, logl): (TKFSimulationResult<MASA>, f64) =
                simulator.simulate_with_fragments_and_logl(&mut rng);
            let alignment = result.phylo.msa;
            assert_eq!(
                alignment.seq_count() + alignment.ancestral_seqs().len(),
                tree.len()
            );
            let phylo = PhyloInfo {
                msa: alignment.clone(),
                tree,
            };
            assert!(
                phylo.check_dollos_constraint().is_ok(),
                "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
            );
            let cost = TKF92FixedIndelCostBuilder::new(
                &[lambda, mu, r],
                result.fragmentation.right_exclusive_boundaries().to_vec(),
                phylo,
            )
            .build()
            .unwrap()
            .logl();
            assert_relative_eq!(logl, cost, epsilon = 1e-10);
        }
    }

    #[test]
    fn tkf91_indel_simulate() {
        for seed in 0..100 {
            let tkf_model = TKF91IndelModel::default();
            let lambda = tkf_model.lambda();
            let mu = tkf_model.mu();
            let mut tree = tree!("(((9: 0.220067, (10: 0.150408, 6: 0.150408): 0.069660): 0.054476, (5: 0.033214, 8: 0.033214): 0.241329): 0.725457, ((((3: 0.033362, 1: 0.033362): 0.000893, 7: 0.034255): 0.379610, 4: 0.413865): 0.310280, 2: 0.724145): 0.275855);");
            set_missing_tree_node_ids(&mut tree).unwrap();

            // Should be so large that it doesn't kick in. Because otherwise the
            // accumulated logl would not match the clean cost. Since event probabilities
            // are summed together for the case of drawing a max_insertion_length.
            let max_insertion_length = Some(100);
            let simulator =
                TKFIndelMSASimulator::new(tkf_model, tree.clone(), max_insertion_length).unwrap();
            let mut rng = DefaultGenerator::new(seed);
            let (result, logl): (TKFSimulationResult<MASA>, f64) =
                simulator.simulate_with_fragments_and_logl(&mut rng);
            let alignment = result.phylo.msa;
            assert_eq!(alignment.seq_count() + alignment.ancestral_seqs().len(), 19);

            // In TKF91 every residue is its own independent link (fragment length == 1), so the
            // fragmentation must be exactly [1, 2, 3, ..., n_cols].
            let n_cols = result.fragmentation.len();
            let expected_fragmentation: Vec<usize> = (1..=n_cols).collect();
            assert_eq!(
                result.fragmentation.right_exclusive_boundaries(),
                expected_fragmentation,
                "TKF91 fragmentation must have every column as its own block"
            );

            // The simulation logl must equal the TKF91 indel cost for the same alignment.
            let phylo = PhyloInfo {
                msa: alignment.clone(),
                tree,
            };
            assert!(
                phylo.check_dollos_constraint().is_ok(),
                "Simulated alignment must satisfy Dollo's constraint (no re-gain of characters)"
            );
            let cost = TKF91IndelCostBuilder::new(&[lambda, mu], phylo)
                .build()
                .unwrap()
                .logl();
            assert_relative_eq!(logl, cost, epsilon = 1e-10);
        }
    }

    #[test]
    fn tkf92_indel_sim_consistency() {
        let lambda = 0.29;
        let mu = 0.3;
        let r = 0.9;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = Some(2);
        let seed = 123;
        let simulator1 =
            TKFIndelMSASimulator::new(tkf_model.clone(), tree.clone(), max_len).unwrap();
        let simulator2 = TKFIndelMSASimulator::new(tkf_model, tree, max_len).unwrap();
        let mut rng1 = DefaultGenerator::new(seed);
        let mut rng2 = DefaultGenerator::new(seed);
        let result1 = simulator1.simulate_with_fragments::<MASA, _>(&mut rng1);
        let msa2: MASA = simulator2.simulate_ancestral_alignment(&mut rng2).msa;
        assert_eq!(result1.phylo.msa.to_string(), msa2.to_string());
    }

    #[test]
    fn build_root_links_no_zero_length_links() {
        let tkf_model = TKF92IndelModel::new(0.01, 0.0105, 0.7);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = Some(50);
        for seed in 0..1000 {
            let mut simulator =
                TKFIndelMSASimulator::new(tkf_model.clone(), tree.clone(), max_len).unwrap();
            simulator.root_length(RootLength::Expected);
            let mut rng = DefaultGenerator::new(seed);
            let root_links = simulator.build_root_links(&mut rng);
            for (i, link) in root_links.iter().skip(1).enumerate() {
                assert!(
                    link.length > 0,
                    "seed {seed}: zero-length root link at index {}",
                    i + 1
                );
            }
        }
    }

    #[test]
    fn tkf92_indel_capping() {
        let lambda = 1.00;
        let mu = 1.1;
        let r = 0.5;
        let tkf_model = TKF92IndelModel::new(lambda, mu, r);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        let max_len = Some(0); // Cap at 0 insertions on branches
        let mut simulator = TKFIndelMSASimulator::new(tkf_model, tree.clone(), max_len).unwrap();
        simulator.root_length(RootLength::Defined(10000));
        let mut rng = DefaultGenerator::new(123);
        let result = simulator.simulate_with_fragments::<MASA, _>(&mut rng);
        let msa = result.phylo.msa;
        // With max_len = 0, no insertions can happen on branches.
        // Thus, every column in the MSA must have a char at the root.
        let root_map = msa.ancestral_map(&tree.root);
        for (col_idx, site) in root_map.iter().enumerate() {
            assert!(
                site.is_some(),
                "Column {} should have a character at the root",
                col_idx
            );
        }
    }

    #[test]
    fn tkf92_indel_capping_hits_cap_for_all_fates() {
        let tkf_model = TKF92IndelModel::new(0.19, 0.2, 0.5);
        let tree = tree!("(A:1.0,B:1.0)R:1.0;");
        // Cap at 0 inserted fragments, so both fates always fall through to the capping branch.
        let simulator = TKFIndelMSASimulator::new(tkf_model, tree, Some(0)).unwrap();

        // A uniform sample of 1.0 can never be smaller than a cumulative probability, so the
        // sampling loops are exhausted and the insertion length is capped.
        assert!(matches!(
            simulator.sample_homolog_fate(1.0, 1.0),
            LinkFate::Homolog(0)
        ));
        assert!(matches!(
            simulator.sample_non_homolog_fate(1.0, 1.0),
            LinkFate::NonHomolog(0)
        ));
    }
}
