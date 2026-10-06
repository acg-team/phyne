use pest::error::Error as PestError;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum Error {
    #[error("IO error: {0}")]
    Io(String),

    #[error("Alphabet error: {0}")]
    Alphabet(String),

    #[error("Sequence error: {0}")]
    Sequence(String),

    #[error("Substitution model error: {0}")]
    SubstitutionModel(#[from] SubstitutionModelError),

    #[error("Evolutionary model error: {0}")]
    EvolutionaryModel(#[from] EvolutionaryModelError),

    #[error("Alignment error: {0}")]
    Alignment(String),

    #[error("Ancestral alignment error: {0}")]
    AncestralAlignment(String),

    #[error("Edge sequence re-estimation error: {0}")]
    EdgeSeqsReestimator(String),

    #[error("Tree error: {0}")]
    Tree(String),

    #[error("Tree move error: {0}")]
    TreeMove(String),

    #[error("Tree parsing error: {0}\n{1}")]
    TreeParsing(
        String,
        #[source] Box<PestError<crate::tree::tree_parser::Rule>>,
    ),

    // Wrapper for external errors
    #[error(transparent)]
    Other(#[from] anyhow::Error),
}

impl From<std::io::Error> for Error {
    fn from(err: std::io::Error) -> Self {
        Error::Io(err.to_string())
    }
}

#[derive(Error, Debug)]
pub enum SubstitutionModelError {
    #[error("Other substitution model error: {0}")]
    Other(String),

    #[error("{name}: expected {expected} frequencies, got {actual}")]
    FrequencyCount {
        name: String,
        expected: usize,
        actual: usize,
    },

    #[error("{name}: frequencies must sum to 1.0")]
    FrequencySum { name: String },

    #[error("{name}: one or more frequency values is negative")]
    NegativeFrequency { name: String },

    #[error("{name}: expects equal frequencies")]
    UnequalFrequencies { name: String },

    #[error("{name}: expected {expected} parameter values, got {actual}")]
    ParameterCount {
        name: String,
        expected: usize,
        actual: usize,
    },
}

#[derive(Error, Debug)]
pub enum EvolutionaryModelError {
    #[error("Other evolutionary model error: {0}")]
    Other(String),

    #[error("{name}: expected {expected} parameter values, got {actual}")]
    ParameterCount {
        name: String,
        expected: usize,
        actual: usize,
    },
}
