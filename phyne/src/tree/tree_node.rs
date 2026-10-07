use std::fmt::{Debug, Display};

use approx::relative_eq;

use crate::tree::NodeIdx::{self, Internal as Int, Leaf};

#[derive(Clone)]
pub struct Node {
    pub idx: NodeIdx,
    pub parent: Option<NodeIdx>,
    pub children: Vec<NodeIdx>,
    pub blen: f64,
    pub id: String,
}

impl Display for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.id.is_empty() {
            write!(f, "{}", self.idx)
        } else {
            write!(f, "{} with id {}", self.idx, self.id)
        }
    }
}

impl Debug for Node {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.id.is_empty() {
            writeln!(
                f,
                "{:?}:{}, parent: {:?}, children: {:?}",
                self.idx, self.blen, self.parent, self.children,
            )
        } else {
            writeln!(
                f,
                "({}) {:?}:{}, parent: {:?}, children: {:?}",
                self.id, self.idx, self.blen, self.parent, self.children,
            )
        }
    }
}

impl PartialEq for Node {
    fn eq(&self, other: &Self) -> bool {
        (self.idx == other.idx)
            && (self.parent == other.parent)
            && (self.children.iter().min() == other.children.iter().min())
            && (self.children.iter().max() == other.children.iter().max())
            && relative_eq!(self.blen, other.blen)
    }
}

impl Node {
    pub(crate) fn new_leaf(idx: usize, parent: Option<NodeIdx>, blen: f64, id: String) -> Self {
        Self {
            idx: Leaf(idx),
            parent,
            children: Vec::new(),
            blen,
            id,
        }
    }

    pub(crate) fn new_internal(
        idx: usize,
        parent: Option<NodeIdx>,
        children: Vec<NodeIdx>,
        blen: f64,
        id: String,
    ) -> Self {
        Self {
            idx: Int(idx),
            parent,
            children,
            blen,
            id,
        }
    }
}
