// run with
// cargo bench --bench decomposition_benchmark -- baseline
// cargo bench --bench decomposition_benchmark -- spectral
// on version you'd like to benchmark

use phylo::evolutionary_models::EvoModel;
use phylo::substitution_models::{SubstModel, WAG};
use std::env;
use std::fs::OpenOptions;
use std::hint::black_box;
use std::io::{self, Write};
use std::time::Instant;

fn main() -> io::Result<()> {
    let label = env::args()
        .nth(1)
        .unwrap_or_else(|| "unlabelled".to_owned());
    let times = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0];
    let rounds = 1_000;
    let model = SubstModel::<WAG>::default();

    // Warm up the same model that will be used for every timed call.
    for &t in &times {
        black_box(model.p(black_box(t)));
    }

    let mut samples = Vec::new();
    let calls = rounds * times.len();
    for _ in 0..7 {
        let start = Instant::now();
        for _ in 0..rounds {
            for &t in &times {
                black_box(model.p(black_box(t)));
            }
        }
        samples.push(start.elapsed().as_secs_f64() * 1e6 / calls as f64);
    }
    samples.sort_by(f64::total_cmp);
    let median = samples[samples.len() / 2];

    let mut csv = OpenOptions::new()
        .create(true)
        .append(true)
        .open("bench_p.csv")?;
    if csv.metadata()?.len() == 0 {
        writeln!(
            csv,
            "label,model,calls_per_sample,samples,median_us_per_call"
        )?;
    }
    let escaped_label = label.replace('"', "\"\"");
    writeln!(
        csv,
        "\"{escaped_label}\",WAG,{calls},{},{median:.6}",
        samples.len()
    )?;
    println!("{label}: {median:.3} microseconds per call (saved to bench_p.csv)");
    Ok(())
}
