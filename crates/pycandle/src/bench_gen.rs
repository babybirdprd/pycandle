use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub fn generate_benchmark(
    model_name: &str,
    manifest_path: PathBuf,
    output_path: PathBuf,
) -> Result<()> {
    println!("🚀 Generating benchmark for {}...", model_name);

    // 1. Read manifest
    let manifest_content =
        fs::read_to_string(&manifest_path).context("Failed to read manifest file")?;

    // Minimal manifest structure
    #[derive(serde::Deserialize)]
    struct Manifest {
        #[serde(flatten)]
        _layers: HashMap<String, serde_json::Value>,
    }
    let _full_manifest: Manifest = serde_json::from_str(&manifest_content)?;

    // We just need to know the inputs to generate dummy data
    // For now, we'll assume a standard 'forward' signature with 'xs: &Tensor'
    // or handle the 'xs' placeholder from the manifest.

    // 2. Generate content
    let bench_code = format!(
        r#"use criterion::{{criterion_group, criterion_main, Criterion}};
use candle_core::{{Device, Tensor, DType}};
use candle_nn::VarBuilder;
use pycandle_core::layers::*; // Ensure project library exposes this
use my_project::{{{}, Config}}; // Adjust crate name if needed

fn criterion_benchmark(c: &mut Criterion) {{
    let device = Device::new_cuda(0).unwrap_or(Device::Cpu);
    println!("Benchmarking on device: {{:?}}", device);

    // Dummy Weights
    let vb = VarBuilder::zeros(DType::F32, &device);
    let config = Config::default();
    
    // Load Model
    let model = {}::load(vb, config).expect("Failed to load model");

    // Dummy Input (Adjust shape as needed)
    let input_shape = (1, 128); // Batch 1, Seq 128
    let input = Tensor::randn(0f32, 1f32, input_shape, &device).unwrap();

    c.bench_function("forward_pass", |b| {{
        b.iter(|| {{
            let _ = model.forward(&input).unwrap();
        }})
    }});
}}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
"#,
        model_name, model_name
    );

    // 3. Write to file
    if let Some(parent) = output_path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output_path, bench_code)?;
    println!("✅ Benchmark generated at {:?}", output_path);
    println!("👉 Run with: cargo bench");

    Ok(())
}
