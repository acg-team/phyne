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
    SubstitutionModel(String),

    #[error("Evolutionary model error: {0}")]
    EvolutionaryModel(String),

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
