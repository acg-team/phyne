use rand::{RngCore, SeedableRng};

/// A fake random number generator for deterministic testing.
/// Can return pre-configured values for unsigned (usize, u64, u32, u16, u8) and
/// signed (isize, i64, i32, i16, i8) integer types.
/// Can also be set up to produce specific f64 values (up to floating point precision).
/// If no pre-configured values are provided, it will return 0 for all integer types and 0.0 for f64.
/// Pre-configured values deterministically influece the order if shuffle is called.
/// If a no-shuffle is desired, provide u64::MAX as the pre-configured value.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FakeRng {
    u64_values: Vec<u64>,
    u64_index: usize,
}

impl RngCore for FakeRng {
    fn next_u32(&mut self) -> u32 {
        self.next_u64_value() as u32
    }

    fn next_u64(&mut self) -> u64 {
        self.next_u64_value()
    }

    fn fill_bytes(&mut self, dest: &mut [u8]) {
        // Fill the byte array with values from the saved u64 sequence
        for chunk in dest.chunks_mut(8) {
            let value = self.next_u64_value();
            let bytes = value.to_le_bytes();
            for (i, byte) in chunk.iter_mut().enumerate() {
                *byte = bytes[i];
            }
        }
    }
}

// Implement SeedableRng but the seed is ignored since this is a fake RNG
impl SeedableRng for FakeRng {
    type Seed = [u8; 8];

    fn from_seed(_: Self::Seed) -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }

    fn seed_from_u64(_: u64) -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }
}

impl FakeRng {
    /// Create a new FakeRng with an empty u64 value sequence, will return 0 for every int value
    pub fn new() -> Self {
        Self {
            u64_values: Vec::new(),
            u64_index: 0,
        }
    }

    /// Create a FakeRng with pre-configured u64 values
    pub fn from_u64_values(values: Vec<u64>) -> Self {
        Self {
            u64_values: values,
            u64_index: 0,
        }
    }
    /// Create a FakeRng that produces specific f64 values using the same transformation of values as rand's
    /// Standard distribution
    pub fn from_f64_values(values: Vec<f64>) -> Self {
        let u64_values = values
            .iter()
            .map(|&f| Self::f64_to_u64_for_rand(f))
            .collect::<Vec<u64>>();
        Self {
            u64_values,
            u64_index: 0,
        }
    }

    /// Convert f64 to u64 using the same algorithm as rand's Standard distribution
    fn f64_to_u64_for_rand(f: f64) -> u64 {
        let clamped = f.clamp(0.0, 1.0 - f64::EPSILON);
        let scaled = clamped * (1u64 << 53) as f64;
        (scaled as u64) << 11
    }

    /// Convert u64 to f64 using the same algorithm as rand's Standard distribution
    #[cfg(test)]
    fn u64_to_f64_for_rand(u: u64) -> f64 {
        // This mirrors the rand crate's conversion: (u64 >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
        ((u >> 11) as f64) * (1.0 / ((1u64 << 53) as f64))
    }

    /// Get the next u64 value, default is 0
    fn next_u64_value(&mut self) -> u64 {
        if self.u64_values.is_empty() {
            0
        } else {
            let value = self.u64_values[self.u64_index % self.u64_values.len()];
            self.u64_index += 1;
            value
        }
    }
}

impl Default for FakeRng {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[cfg_attr(coverage, coverage(off))]
mod tests {
    use approx::assert_relative_eq;
    use rand::{distr::weighted::WeightedIndex, RngCore, SeedableRng};

    use crate::random::{FakeGenerator, RandomGenerator};

    use super::FakeRng;

    #[test]
    fn fake_rng_defaults() {
        // Test FakeRng defaults
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::default());
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.random();
        assert_eq!(val, 0);
        let val: f64 = fake_rng.random();
        assert_eq!(val, 0.0);
        let val: bool = fake_rng.random();
        assert!(!val);
    }

    #[test]
    fn fake_rng_creation() {
        // Test new FakeGenerator creation with and without seed (seed makes no difference)
        let fake1 = RandomGenerator::from_rng(FakeRng::seed_from_u64(42));
        let fake2 = RandomGenerator::from_rng(FakeRng::from_seed([42; 8]));
        let fake3 = RandomGenerator::from_rng(FakeRng::new());
        let fake4 = RandomGenerator::from_rng(FakeRng::default());
        let fake5 = FakeGenerator::default();
        assert_eq!(fake1.seed(), 0);
        assert_eq!(fake2.seed(), 0);
        assert_eq!(fake3.seed(), 0);
        assert_eq!(fake4.seed(), 0);
        assert_eq!(fake5.seed(), 0);
        assert_eq!(fake1, fake2);
        assert_eq!(fake1, fake3);
        assert_eq!(fake1, fake4);
        assert_eq!(fake1, fake5);
    }

    #[test]
    fn fake_rng_with_u64_values() {
        // FakeRng with pre-configured values
        let values = (15..25).collect::<Vec<u64>>();
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        assert_eq!(fake_rng.seed(), 0);
        for i in 0..30 {
            let val: u64 = fake_rng.random();
            assert_eq!(val, values[i % values.len()]);
        }
        let val: f64 = fake_rng.random();
        assert_eq!(val, FakeRng::u64_to_f64_for_rand(values[0]));
    }

    #[test]
    fn fake_rng_with_diff_types() {
        // FakeRng with different value types
        let values = vec![5, 6, 7, 8, 9, 14, 15, 16, u64::MAX];
        let mut fake_rng = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        assert_eq!(fake_rng.seed(), 0);
        let val: u64 = fake_rng.random();
        assert_eq!(val, values[0]);
        let val: u32 = fake_rng.random();
        assert_eq!(val, values[1] as u32);
        let val: u16 = fake_rng.random();
        assert_eq!(val, values[2] as u16);
        let val: u8 = fake_rng.random();
        assert_eq!(val, values[3] as u8);
        let val: i64 = fake_rng.random();
        assert_eq!(val, values[4] as i64);
        let val: i32 = fake_rng.random();
        assert_eq!(val, values[5] as i32);
        let val: i16 = fake_rng.random();
        assert_eq!(val, values[6] as i16);
        let val: i8 = fake_rng.random();
        assert_eq!(val, values[7] as i8);
        let val: f64 = fake_rng.random();
        assert_eq!(val, FakeRng::u64_to_f64_for_rand(values[8]));
    }

    #[test]
    fn fake_rng_reproducibility() {
        // creating new instances with the same values produces the same sequence
        let values = vec![5, 7, 8, 10, 12];
        let mut rng1 = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        let val1: u64 = rng1.random();
        let val2: u32 = rng1.random();
        let val3: f64 = rng1.random();

        let mut rng2 = RandomGenerator::from_rng(FakeRng::from_u64_values(values.clone()));
        let val1_repeat: u64 = rng2.random();
        let val2_repeat: u32 = rng2.random();
        let val3_repeat: f64 = rng2.random();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val2, val2_repeat);
        assert_eq!(val3, val3_repeat);
        assert_eq!(val1, values[0]);
        assert_eq!(val2, values[1] as u32);
        assert_eq!(val3, FakeRng::u64_to_f64_for_rand(values[2]));
    }

    #[test]
    fn fake_reseed_empty() {
        // reseeding does not do anything to an empty FakeGenerator
        let mut fake_rng = FakeGenerator::default();
        assert_eq!(fake_rng.seed(), 0);
        let val1: u64 = fake_rng.random();

        fake_rng.reseed(42);
        assert_eq!(fake_rng.seed(), 42);
        let val1_repeat: u64 = fake_rng.random();

        assert_eq!(val1, val1_repeat);
        assert_eq!(val1, 0); // Default value after reseed
    }

    #[test]
    fn fake_shuffle() {
        let mut rng = FakeGenerator::default();
        let mut vec = vec![1, 2, 3, 4, 5];
        rng.shuffle(&mut vec);
        assert_eq!(vec, vec![5, 1, 2, 3, 4]);
    }

    #[test]
    fn fake_shuffle_umax() {
        let mut rng = FakeGenerator::from_rng(FakeRng::from_u64_values(vec![u64::MAX]));
        let mut vec = (1..100).collect::<Vec<u64>>();
        rng.shuffle(&mut vec);
        assert_eq!(vec, (1..100).collect::<Vec<u64>>());
    }

    #[test]
    fn fake_sample_default() {
        // always returns first non-zero index when no pre-configured values
        let mut rng = FakeGenerator::default();
        let dist = WeightedIndex::new([1.0, 2.0, 3.0]).unwrap();
        assert_eq!(rng.sample(&dist), 0);
    }

    #[test]
    fn fake_sample() {
        // always returns first non-zero index when no pre-configured values
        let mut rng = FakeGenerator::default();
        let dist = WeightedIndex::new([0.0, 3.0, 2.0, 1.0, 2.0]).unwrap();
        assert_eq!(rng.sample(&dist), 1);
        assert_eq!(rng.sample(&dist), 1);
        assert_eq!(rng.sample(&dist), 1);
    }

    #[cfg(test)]
    fn get_cumulative_index(weights: &[f64], value: f64) -> usize {
        let total: f64 = weights.iter().sum();
        let mut cumulative = 0.0;
        for (i, &weight) in weights.iter().enumerate() {
            cumulative += weight / total;
            if value < cumulative {
                return i;
            }
        }
        weights.len() - 1
    }

    #[test]
    fn fake_sample_with_values() {
        // sampling with pre-configured float values
        let floats = vec![
            0.1,
            0.25,
            0.5,
            0.75,
            0.9,
            0.99,
            0.2 - f64::EPSILON,
            0.2 + f64::EPSILON,
            1.0,
            0.0,
        ];
        let mut rng = RandomGenerator::from_rng(FakeRng::from_f64_values(floats.clone()));
        let weights = vec![0.2, 0.35, 0.4, 0.05];
        let dist = WeightedIndex::new(weights.clone()).unwrap();
        for value in &floats {
            let expected = get_cumulative_index(&weights, *value);
            assert_eq!(rng.sample(&dist), expected);
        }
    }

    #[test]
    fn fake_from_desired_floats() {
        // Fake RNG that produces specific f64 values
        let floats = vec![0.1, 0.25, 0.5, 0.75, 0.9];
        let mut rng = RandomGenerator::from_rng(FakeRng::from_f64_values(floats.clone()));
        for expected in floats {
            assert_relative_eq!(rng.random::<f64>(), expected);
        }
    }

    #[test]
    fn test_f64_u64_roundtrip() {
        // f64 -> u64 -> f64 conversion preserves values within floating point precision
        let test_values = vec![0.0, 0.1, 0.25, 0.333333, 0.5, 0.666666, 0.75, 0.9, 0.999999];

        for original in test_values {
            let u64_val = FakeRng::f64_to_u64_for_rand(original);
            let roundtrip = FakeRng::u64_to_f64_for_rand(u64_val);
            assert_relative_eq!(original, roundtrip, epsilon = 1e-15);
        }
    }

    #[test]
    fn test_u64_to_f64_conversion() {
        // specific u64 values produce expected f64 values
        assert_relative_eq!(FakeRng::u64_to_f64_for_rand(0), 0.0);
        assert_relative_eq!(FakeRng::u64_to_f64_for_rand(u64::MAX), 1.0 - f64::EPSILON);

        // middle value
        let mid_u64 = 1u64 << 63; // Half of the maximum value when shifted
        let mid_f64 = FakeRng::u64_to_f64_for_rand(mid_u64);
        assert_relative_eq!(mid_f64, 0.5, epsilon = 1e-15);
    }

    #[test]
    fn test_fill_bytes() {
        let values = vec![0x1122334455667788, 0x99AABBCCDDEEFF00];
        let mut fake_rng = FakeRng::from_u64_values(values.clone());
        let mut buffer = [0u8; 16];
        fake_rng.fill_bytes(&mut buffer);

        let expected_bytes: Vec<u8> = values
            .iter()
            .flat_map(|&v| v.to_le_bytes().to_vec())
            .collect();

        assert_eq!(buffer.to_vec(), expected_bytes);
    }
}
