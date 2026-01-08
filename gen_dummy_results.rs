use serde::Serialize;
use std::fs::OpenOptions;
use std::io::Write;

#[derive(Serialize)]
struct Result {
    name: String,
    mse: f32,
    max_diff: f32,
    cosine_sim: f32,
    passed: bool,
}

fn main() {
    let results = vec![
        Result {
            name: "layer1.out.0".to_string(),
            mse: 1e-10,
            max_diff: 1e-5,
            cosine_sim: 1.0,
            passed: true,
        },
        Result {
            name: "layer2.out.0".to_string(),
            mse: 1e-8,
            max_diff: 1e-4,
            cosine_sim: 0.9999,
            passed: true,
        },
        Result {
            name: "layer3.out.0".to_string(),
            mse: 0.05,
            max_diff: 0.1,
            cosine_sim: 0.95,
            passed: false,
        }, // Drift!
        Result {
            name: "layer4.out.0".to_string(),
            mse: 0.2,
            max_diff: 0.5,
            cosine_sim: 0.8,
            passed: false,
        },
    ];

    let mut file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open("verification_results.jsonl")
        .unwrap();

    for r in results {
        writeln!(file, "{}", serde_json::to_string(&r).unwrap()).unwrap();
    }

    println!("Generated verification_results.jsonl");
}
