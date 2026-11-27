use std::fmt::Display;
use std::ops::{BitAnd, BitOr, Deref, Sub};

use hashbrown::{hash_set::IntoIter, HashSet};
use itertools::join;

use crate::alphabets::GAP;

/// The `ParsimonySet` struct represents a set of possible characters, e.g. in the Fitch algorithm.
/// It is used to keep track of all potential character states at a given position in a sequence alignment.
#[repr(transparent)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParsimonySet {
    /// The underlying set of characters as a `hashbrown::HashSet<u8>`.
    s: HashSet<u8>,
}

impl ParsimonySet {
    /// Creates a new empty parsimony set.
    pub(crate) fn empty() -> Self {
        Self { s: HashSet::new() }
    }

    /// Creates a new parsimony set containing only the gap character.
    pub(crate) fn gap() -> Self {
        Self {
            s: HashSet::from_iter([GAP]),
        }
    }

    /// Creates a new parsimony set from a slice of characters. Cannot be used to create valid sets with ambiguous
    /// symbols since their interpretation depends on the underlying alphabet and cannot be determined solely from
    /// the slice.
    pub(super) fn from_slice(slice: &[u8]) -> Self {
        Self {
            s: HashSet::from_iter(slice.iter().copied()),
        }
    }
}

impl Deref for ParsimonySet {
    type Target = HashSet<u8>;

    fn deref(&self) -> &Self::Target {
        &self.s
    }
}

impl Display for ParsimonySet {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut chars: Vec<char> = self.s.iter().map(|&a| a as char).collect();
        chars.sort();
        write!(f, "[{}]", join(chars.iter(), ""))
    }
}

impl IntoIterator for ParsimonySet {
    type Item = u8;
    type IntoIter = IntoIter<u8>;

    fn into_iter(self) -> Self::IntoIter {
        self.s.into_iter()
    }
}

impl BitAnd for &ParsimonySet {
    type Output = ParsimonySet;

    fn bitand(self, rhs: Self) -> Self::Output {
        ParsimonySet {
            s: self.s.intersection(&rhs.s).copied().collect(),
        }
    }
}

impl BitOr for &ParsimonySet {
    type Output = ParsimonySet;

    fn bitor(self, rhs: Self) -> Self::Output {
        ParsimonySet {
            s: self.s.union(&rhs.s).copied().collect(),
        }
    }
}

impl Sub for &ParsimonySet {
    type Output = ParsimonySet;

    fn sub(self, rhs: Self) -> Self::Output {
        ParsimonySet {
            s: self.s.difference(&rhs.s).copied().collect(),
        }
    }
}
