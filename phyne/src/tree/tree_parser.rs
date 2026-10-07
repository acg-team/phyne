use fixedbitset::FixedBitSet;
use hashbrown::HashSet;
use log::{info, warn};
use pest::{iterators::Pair, Parser};
use pest_derive::Parser;

use crate::bail;
use crate::tree::{
    Node,
    NodeIdx::{self, Internal as Int, Leaf},
    Tree,
};
use crate::Result;

#[derive(Parser)]
#[grammar = "./tree/newick.pest"]
pub struct NewickGrammar;

/// Parses Newick formatted trees from a string into a vector of `Tree` structures.
///
/// Only binary trees (rooted or unrooted) are supported, with each internal node having exactly two children.
/// Unrooted trees are automatically rooted at the trifurcation node.
/// Leaf ids are required to be unique and must start with a letter (a-z, A-Z). Duplicate leaf ids will result in an error.
/// Internal node ids are optional but should also be unique and start with a letter (a-z, A-Z) if provided.
pub fn from_newick(newick: &str) -> Result<Vec<Tree>> {
    NewickTreeParser::new().parse(newick)
}

#[derive(Debug, Clone, Copy, Default)]
pub struct NewickTreeParser {}

impl NewickTreeParser {
    /// Create a new instance of the NewickTreeParser.
    pub fn new() -> Self {
        Self::default()
    }

    /// Parse a Newick formatted string into a vector of `Tree` structures.
    ///
    /// ```
    /// use phylo::tree::tree_parser::from_newick;
    ///
    /// let newick = "(A:0.1,B:0.2);";
    /// let trees = from_newick(newick).unwrap();
    /// assert_eq!(trees.len(), 1);
    /// ```
    pub fn parse(&self, newick: &str) -> Result<Vec<Tree>> {
        info!("Parsing newick trees");
        let mut trees = Vec::new();
        let newick_tree_res = NewickGrammar::parse(Rule::newick, newick);
        if let Err(e) = newick_tree_res {
            bail!(TreeParsing, "malformed newick string", Box::new(e));
        }

        let newick_tree_rule = newick_tree_res.unwrap().next().unwrap();
        match newick_tree_rule.as_rule() {
            Rule::newick => {
                for tree_rule in newick_tree_rule.into_inner() {
                    if let Some(rule) = tree_rule.into_inner().next() {
                        trees.push(TreeBuilder::parse_tree(rule)?);
                    }
                }
            }
            _ => unreachable!(),
        }
        info!("Finished parsing newick trees successfully");
        Ok(trees)
    }
}

/// Mutable, tree-local build state for a single Newick parse.
///
/// The parser remains reusable and stateless; each tree gets its own builder so leaf
/// validation and node numbering are scoped to that tree alone.
struct TreeBuilder {
    tree: Tree,
    leaf_ids: HashSet<String>,
    node_idx: usize,
    parent_stack: Vec<usize>,
}

impl TreeBuilder {
    /// Create a new TreeBuilder with an empty tree and null state.
    fn new() -> Self {
        Self {
            tree: Self::empty_tree(),
            leaf_ids: HashSet::new(),
            node_idx: 0,
            parent_stack: Vec::new(),
        }
    }

    /// Create a new empty tree with no nodes or orderings.
    fn empty_tree() -> Tree {
        Tree {
            root: Int(0),
            nodes: Vec::new(),
            postorder: Vec::new(),
            preorder: Vec::new(),
            n: 0,
            length: 0.0,
            dirty: FixedBitSet::new(),
        }
    }

    /// Parse a single tree rule into a `Tree` structure.
    fn parse_tree(tree_rule: Pair<Rule>) -> Result<Tree> {
        let mut builder = Self::new();

        match tree_rule.as_rule() {
            Rule::rooted => builder.parse_rooted_tree(tree_rule)?,
            Rule::unrooted => builder.parse_unrooted_tree(tree_rule)?,
            _ => unreachable!(),
        }

        builder.tree.finalise();
        Ok(builder.tree)
    }

    /// Parse a rooted tree rule, which can be either a leaf or an internal node with children.
    fn parse_rooted_tree(&mut self, tree_rule: Pair<Rule>) -> Result<()> {
        let node_rule = tree_rule.into_inner().next().unwrap();

        match node_rule.as_rule() {
            Rule::leaf => {
                self.tree.root = Leaf(self.node_idx);
                let id = self.parse_leaf(node_rule)?;
                self.verify_leaf_id(id)?;
            }
            Rule::internal => {
                self.tree.root = Int(self.node_idx);
                self.parse_internal(node_rule)?;
            }
            _ => unreachable!(),
        }

        Ok(())
    }

    /// Parse an unrooted tree rule, which is expected to have exactly three children and will be rooted at the trifurcation.
    fn parse_unrooted_tree(&mut self, tree_rule: Pair<Rule>) -> Result<()> {
        warn!("Found unrooted tree, will root at the trifurcation");

        let mut children: Vec<NodeIdx> = Vec::new();
        for node_rule in tree_rule.into_inner() {
            self.append_child(node_rule, &mut children)?;
        }

        self.root_unrooted_tree_at_trifurcation(children);
        Ok(())
    }

    /// Append a child node to the current list of children, updating the builder's state accordingly.
    fn append_child(&mut self, node_rule: Pair<Rule>, children: &mut Vec<NodeIdx>) -> Result<()> {
        match node_rule.as_rule() {
            Rule::leaf => {
                children.push(Leaf(self.node_idx));
                let id = self.parse_leaf(node_rule)?;
                self.verify_leaf_id(id)?;
            }
            Rule::internal => {
                children.push(Int(self.node_idx));
                self.parse_internal(node_rule)?;
            }
            _ => unreachable!(),
        }
        Ok(())
    }

    /// Verify that the leaf id has not yet been seen in the tree. If it is a duplicate, return an error.
    fn verify_leaf_id(&mut self, id: String) -> Result<()> {
        if !self.leaf_ids.insert(id.clone()) {
            bail!(Tree, "duplicate leaf id ({}) found in the tree", id);
        }
        Ok(())
    }

    /// Parse an internal node rule, which may contain a label, branch length, and child nodes (leaves or internal node).
    fn parse_internal(&mut self, internal_node: Pair<Rule>) -> Result<()> {
        let mut id = String::from("");
        let mut blen = 0.0;
        let mut children: Vec<NodeIdx> = Vec::new();

        self.parent_stack.push(self.node_idx);

        self.tree.nodes.push(Node::new_internal(
            self.node_idx,
            None,
            vec![],
            0.0,
            "".to_string(),
        ));

        self.node_idx += 1;
        for rule in internal_node.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label(rule),
                Rule::branch_length => blen = Self::parse_branch_length(rule),
                Rule::internal | Rule::leaf => {
                    self.append_child(rule, &mut children)?;
                }
                _ => unreachable!(),
            }
        }
        let cur_node_idx = self
            .parent_stack
            .pop()
            .expect("newick parser stack underflow error");

        for child_idx in &children {
            self.tree.nodes[usize::from(child_idx)].parent = Some(Int(cur_node_idx));
        }
        self.tree.nodes[cur_node_idx].id = id;
        self.tree.nodes[cur_node_idx].blen = blen;
        self.tree.nodes[cur_node_idx].children = children;
        Ok(())
    }

    /// Parse a leaf node rule, which may contain a label and branch length. Returns the leaf id.
    fn parse_leaf(&mut self, leaf_node: Pair<Rule>) -> Result<String> {
        let mut id = String::from("");
        let mut blen = 0.0;
        for rule in leaf_node.into_inner() {
            match rule.as_rule() {
                Rule::label => id = Self::parse_label(rule),
                Rule::branch_length => blen = Self::parse_branch_length(rule),
                _ => unreachable!(),
            }
        }
        self.tree
            .nodes
            .push(Node::new_leaf(self.node_idx, None, blen, id.clone()));

        self.node_idx += 1;
        Ok(id)
    }

    /// Convert a three-way unrooted split into a rooted binary tree by introducing the
    /// trifurcation node and making it the new root.
    fn root_unrooted_tree_at_trifurcation(&mut self, children: Vec<NodeIdx>) {
        let mut node_idx = self.node_idx;
        let new_children = children[0..2].to_vec();
        for child_idx in new_children.iter() {
            self.tree.nodes[usize::from(child_idx)].parent = Some(Int(node_idx));
        }

        self.tree.nodes.push(Node::new_internal(
            node_idx,
            None,
            new_children,
            0.0,
            "".to_string(),
        ));

        let new_children = vec![Int(node_idx), children[2]];

        node_idx += 1;
        for child_idx in new_children.iter() {
            self.tree.nodes[usize::from(child_idx)].parent = Some(Int(node_idx));
        }

        self.tree.nodes.push(Node::new_internal(
            node_idx,
            None,
            new_children,
            0.0,
            "".to_string(),
        ));
        self.tree.root = Int(node_idx);
        self.node_idx = node_idx + 1;
    }

    /// Parse a branch length rule, which is expected to be a floating-point number.
    fn parse_branch_length(branch_length: Pair<Rule>) -> f64 {
        branch_length
            .into_inner()
            .next()
            .unwrap()
            .as_str()
            .trim()
            .parse::<f64>()
            .unwrap_or_default()
    }

    /// Parse a label rule, which is expected to be a string identifier.
    fn parse_label(label: Pair<Rule>) -> String {
        label.as_str().to_string()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use crate::tree::tree_parser::NewickTreeParser;

    #[test]
    fn test_parser_multiple_parse_calls() {
        let parser = NewickTreeParser::new();

        let str1 = "((((A:1.0,B:1.0)F:1.0,C:2.0)G:1.0,D:3.0)H:1.0,E:4.0)I:1.0;\
            ((A:1.0,B:2.0)E:5.1,(C:3.0,D:4.0)F:6.2)G:7.3;";
        let str2 = "(A:1.0,(B:1.0,C:1.0)E:2.0)F:1.0;";

        let trees1 = parser.parse(str1).unwrap();
        assert_eq!(trees1.len(), 2);

        let trees2 = parser.parse(str2).unwrap();
        assert_eq!(trees2.len(), 1);
    }
}
