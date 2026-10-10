#![cfg_attr(coverage, feature(coverage_attribute))]

// Re-export commonly used types for convenience with macros
pub use bio::io::fasta::Record;

pub mod error;
pub use error::*;

pub mod alignment;
pub mod alphabets;
pub mod asr;
pub mod evolutionary_distances;
pub mod evolutionary_models;
pub mod io;
pub mod likelihood;
pub mod optimisers;
pub mod parsimony;
pub mod parsimony_presence_absence;
pub mod phylo_info;
pub mod pip_model;
pub mod random;
pub mod substitution_models;
pub mod tkf_model;
pub mod tree;

pub(crate) mod macros;

pub type Result<T> = std::result::Result<T, Error>;

pub(crate) const MAX_BLEN: f64 = 1e5f64;

pub(crate) const DEFAULT_EPSILON: f64 = 1e-3;

pub(crate) const REPORT_ISSUES_URL: &str = "https://github.com/acg-team/rust-phylo/issues";
