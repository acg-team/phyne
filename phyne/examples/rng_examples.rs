use rand::distr::weighted::WeightedIndex;
use rand::rngs::StdRng;
use std::iter::repeat_n;

use phylo::random::{DefaultGenerator, RandomGenerator};

fn main() {
    // 1. Generating from default RNG (Pcg32, seeded with the current timestamp):
    let mut rng = DefaultGenerator::default();
    println!(
        "1. Generating from default RNG (Pcg32-based), seeded with the current timestamp {}:",
        rng.seed()
    );
    println!("  Random f64: {:.6}", rng.random::<f64>());
    println!("  Random u32: {}", rng.random::<u32>());
    println!("  Random i32: {}", rng.random::<i32>());
    println!("  Probability [0,1): {:.6}", rng.random::<f64>());
    println!("  Range 1-100: {}", rng.random_range(1..=100));
    println!("  Range 0.0-10.0: {:.3}", rng.random_range(0.0..10.0));
    println!("  Boolean (50%): {}", rng.random_bool(0.5));
    println!("  Boolean (25%): {}", rng.random_bool(0.25));
    let dist = WeightedIndex::new(repeat_n(1.0, 15)).unwrap();
    let sample = rng.sample(&dist);
    println!("  Sample from a uniform weighted distribution: {}", sample);
    println!();

    // 2. Using the default RNG (Pcg32-based)
    let seed = 42;
    let mut rng = DefaultGenerator::new(seed);
    println!(
        "2. Default RNG (Pcg32) with seed {} - reproducible:",
        rng.seed()
    );
    for i in 0..3 {
        println!("  {i}: {:.6}", rng.random::<f64>());
    }
    // Reset and show reproducibility
    rng.reseed(seed);
    println!("After resetting to seed {}:", rng.seed());
    for i in 0..3 {
        println!("  {i}: {:.6}", rng.random::<f64>());
    }
    println!();

    // 3. Using a different seed for another Pcg32 instance
    let seed = 123;
    let mut another_pcg_rng = DefaultGenerator::new(seed);
    println!(
        "3. Another Pcg32 instance with different seed {}:",
        another_pcg_rng.seed()
    );
    for i in 0..3 {
        println!("  {i}: {:.6}", another_pcg_rng.random::<f64>());
    }
    println!();

    // 4. Creating a custom RNG instance with a different generator (StdRng)
    let mut custom_rng = RandomGenerator::<StdRng>::new(seed);
    println!(
        "4. Custom RNG instance with StdRng with seed {}:",
        custom_rng.seed()
    );
    for i in 0..3 {
        println!("  {i}: {:.6}", custom_rng.random::<f64>());
    }
    println!();
}
