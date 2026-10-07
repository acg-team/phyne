use ntimestamp::Timestamp;
use rand::distr::uniform::{SampleRange, SampleUniform};
use rand::distr::{Distribution, StandardUniform};
use rand::prelude::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_pcg::Pcg32;

pub mod fake_random;
pub use fake_random::*;

/// A generic random number generator wrapper that can work with different RNGs.
///
/// This provides a reusable interface for different RNG backends, which must implement `Rng + SeedableRng`.
/// Note: This implementation is not thread-reproducible, i.e. if used in multiple threads, the sequences
/// generated may differ between runs. For phylogenetic analysis reproducibility both sequential and parallel
/// runs should produce the same results when using the same seed.
/// Users must handle this accordingly to ensure reproducibility if needed.
///
/// # Examples
///
/// ```rust
/// use rand::rngs::StdRng;
///
/// use phylo::random::RandomGenerator;
///
/// // Create a custom RNG instance
/// let mut custom_rng: RandomGenerator<StdRng> = RandomGenerator::new(123);
/// let custom_value: f64 = custom_rng.random();
/// ```
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct RandomGenerator<R>
where
    R: Rng + SeedableRng,
{
    pub seed: u64,
    pub rng: R,
}

/// Type alias for the default RNG implementation which uses Pcg32 which is platform-independent.
/// Note: Pcg32 is not the most secure RNG, but is fast and suitable for most phylogenetic applications.
pub type DefaultGenerator = RandomGenerator<Pcg32>;

impl Default for DefaultGenerator {
    fn default() -> Self {
        let seed = Timestamp::now().as_u64(); // Use current timestamp as seed
        Self::new(seed)
    }
}

/// Type alias for a Fake RNG implementation for testing purposes.
pub type FakeGenerator = RandomGenerator<FakeRng>;

impl Default for FakeGenerator {
    fn default() -> Self {
        let seed = 0;
        Self::new(seed)
    }
}

impl<R> RandomGenerator<R>
where
    R: Rng + SeedableRng,
{
    /// Create a new RandomGenerator with the given seed.
    pub fn new(seed: u64) -> Self {
        Self {
            seed,
            rng: R::seed_from_u64(seed),
        }
    }

    #[cfg(test)]
    /// Create a new RandomGenerator from a given RNG instance.
    pub(crate) fn from_rng(rng: R) -> Self {
        Self { seed: 0, rng }
    }

    /// Get the current seed of the RNG.
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Generate a random value of type T.
    pub fn random<T>(&mut self) -> T
    where
        StandardUniform: Distribution<T>,
    {
        self.rng.random()
    }

    /// Generate a random bool with probability p.
    pub fn random_bool(&mut self, p: f64) -> bool {
        self.rng.random_bool(p)
    }

    /// Shuffle a slice in place.
    pub fn shuffle<T>(&mut self, slice: &mut [T]) {
        slice.shuffle(&mut self.rng);
    }

    /// Reseed the RNG with a new seed.
    pub fn reseed(&mut self, seed: u64) {
        *self = Self::new(seed);
    }

    /// Sample from a weighted distribution.
    pub fn sample<D, T>(&mut self, dist: &D) -> T
    where
        D: Distribution<T>,
    {
        self.rng.sample(dist)
    }

    /// Generate a random value in the specified range.
    pub fn random_range<T, Range>(&mut self, range: Range) -> T
    where
        T: SampleUniform,
        Range: SampleRange<T>,
    {
        self.rng.random_range(range)
    }
}

#[cfg_attr(coverage, coverage(off))]
#[cfg(test)]
mod tests {
    use itertools::repeat_n;
    use rand::distr::weighted::WeightedIndex;

    use super::*;

    #[test]
    fn rng_reproducibility() {
        // Test that creating new instances with the same seed produces the same sequence
        let mut rng1 = DefaultGenerator::new(42);
        assert_eq!(rng1.seed(), 42);
        let val1: f64 = rng1.random();
        let val2: u32 = rng1.random();

        let mut rng2 = DefaultGenerator::new(42);
        assert_eq!(rng2.seed(), 42);
        let val1_repeat: f64 = rng2.random();
        let val2_repeat: u32 = rng2.random();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
    }

    #[test]
    fn rng_different_types() {
        // Test that most common types can be generated from the same RNG instance and the
        // generator does not panic.
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let _val: u64 = rng.random();
        let _val: u32 = rng.random();
        let _val: u16 = rng.random();
        let _val: u8 = rng.random();
        let _val: i128 = rng.random();
        let _val: i64 = rng.random();
        let _val: i32 = rng.random();
        let _val: i16 = rng.random();
        let _val: i8 = rng.random();
        let _val: char = rng.random();
        let _val: bool = rng.random();
        let val: f64 = rng.random();
        assert!(val.is_finite() && !val.is_nan());
        let val: f32 = rng.random();
        assert!(val.is_finite() && !val.is_nan());
    }

    #[test]
    fn reseed() {
        // Test that reseeding to the same value produces the same sample
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let val1: f64 = rng.random();

        rng.reseed(42);
        assert_eq!(rng.seed(), 42);
        let val1_repeat: f64 = rng.random();

        assert_eq!(val1, val1_repeat);
    }

    #[test]
    fn rng_functions() {
        let mut rng = DefaultGenerator::new(123);
        assert_eq!(rng.seed(), 123);
        assert!((0.0..1.0).contains(&rng.random::<f64>()));
        assert!((0.0..10.0).contains(&rng.random_range(0.0..10.0)));
        assert!((1..100).contains(&rng.random_range(1..100)));

        let _random_bool = rng.random_bool(0.5);
    }

    #[test]
    fn rng_range() {
        let mut rng = DefaultGenerator::new(123);
        assert_eq!(rng.seed(), 123);
        for _ in 0..10 {
            let val: u32 = rng.random_range(1..100);
            assert!((1..100).contains(&val));
        }
    }

    #[test]
    fn different_seeds_produce_different_values() {
        // This is not guaranteed to produce different values because there are no guarantees on the Rng
        // implementation, but it is extremely unlikely that it will fail.
        let mut rng = DefaultGenerator::new(1);
        assert_eq!(rng.seed(), 1);
        let val1: f64 = rng.random();
        rng.reseed(2);
        assert_eq!(rng.seed(), 2);
        let val1_repeat: f64 = rng.random();
        assert_ne!(val1, val1_repeat);
    }

    #[test]
    fn different_timestamps_produce_different_values() {
        // DefaultGenerator uses the current timestamp as a seed, so different instances should have different seeds
        // per definition of ntimestamp, but not guaranteed to produce different values, as in the previous test.
        let timestamp = Timestamp::now().as_u64();
        let mut rng = DefaultGenerator::default();
        assert!(rng.seed() > timestamp);
        let val1: u64 = rng.random();
        let val2: f64 = rng.random();
        let mut rng2 = DefaultGenerator::default();
        assert!(rng2.seed() > rng.seed());
        assert!(rng2.seed() > timestamp);
        let val1_repeat: u64 = rng2.random();
        let val2_repeat: f64 = rng2.random();
        assert_ne!(val1, val1_repeat);
        assert_ne!(val2, val2_repeat);
    }

    #[test]
    fn shuffle() {
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let mut vec = vec![1, 2, 3, 4, 5];
        let original_vec = vec.clone();
        rng.shuffle(&mut vec);
        assert_ne!(vec, original_vec);
        // Check that all elements are still present
        assert!(vec.iter().all(|x| original_vec.contains(x)));
    }

    #[test]
    fn sample_weighted_index() {
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample < 3);
        let dist = WeightedIndex::new(repeat_n(1.0, 15)).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample < 15);
    }

    #[test]
    fn sample_weighted_single_best() {
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let dist = WeightedIndex::new([1.0, 0.0, 0.0, 0.0, 0.0]).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample == 0);
        let dist = WeightedIndex::new([0.0, 0.0, 1.0, 0.0, 0.0]).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample == 2);
    }

    #[test]
    fn sample_weighted_several_best() {
        let mut rng = DefaultGenerator::new(42);
        assert_eq!(rng.seed(), 42);
        let dist = WeightedIndex::new([0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0]).unwrap();
        let sample = rng.sample(&dist);
        assert!(sample == 2 || sample == 4 || sample == 7);
    }
}
