use std::fmt::Display;

use dyn_clone::DynClone;
use nalgebra::{DMatrix, DVector};

use crate::alphabets::Alphabet;
use crate::Result;

pub type RateMatrix = DMatrix<f64>;
pub type TransitionMatrix = DMatrix<f64>;
pub type FreqVector = DVector<f64>;

#[derive(Clone, clap::ValueEnum, Debug, Copy)]
pub enum FrequencyOptimisation {
    /// The empirical frequencies are calculated from the alignment.
    /// This will not necessarily increase the likelihood.
    Empirical,
    Estimated,
    Fixed,
}

pub trait EvoModel: Display + DynClone {
    fn p(&self, time: f64) -> TransitionMatrix;
    fn q(&self) -> &RateMatrix;
    fn rate(&self, i: u8, j: u8) -> f64;
    fn params(&self) -> &[f64];
    fn set_param(&mut self, param: usize, value: f64);
    fn freqs(&self) -> &FreqVector;
    fn set_freqs(&mut self, pi: FreqVector) -> Result<()>;
    fn n(&self) -> usize;
    fn alphabet() -> &'static Alphabet
    where
        Self: Sized;
}

dyn_clone::clone_trait_object!(EvoModel);
