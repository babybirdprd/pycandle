mod codegen;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use codegen::Codegen;
use pycandle::LayerMeta;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser)]
#[command(name = "pycandle")]
#[command(about = "A tool for bit-perfect parity checking between PyTorch and Candle", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Record activations from a PyTorch model
    Record {
        /// Path to the Python script that defines and runs the model
        #[arg(short, long)]
        script: PathBuf,

        /// Project name for the trace
        #[arg(short, long)]
        name: String,

        /// Output directory for the trace and manifest
        #[arg(short, long, default_value = "pycandle_trace")]
        out: PathBuf,
    },
    /// Generate Candle code from a manifest
    Codegen {
        /// Path to the manifest JSON file
        #[arg(short, long)]
        manifest: PathBuf,

        /// Output path for the generated Rust file
        #[arg(short, long)]
        out: PathBuf,

        /// Name of the model struct to generate
        #[arg(long, default_value = "MyModel")]
        model: String,
    },
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Record { script, name, out } => {
            println!(
                "🚀 Recording trace for project '{}' using script '{:?}'...",
                name, script
            );

            // In a real scenario, we'd want to inject the GoldenRecorder into the user's script
            // or provide a standard way for them to use it.
            // For now, we assume their script already uses spy.py.

            let status = Command::new("uv")
                .arg("run")
                .arg("python")
                .arg(script)
                .spawn()
                .context("Failed to spawn uv run")?
                .wait()
                .context("Failed to wait for python process")?;

            if status.success() {
                println!("✅ Recording complete. Files should be in {:?}", out);
            } else {
                eprintln!("❌ Recording failed.");
            }
        }
        Commands::Codegen {
            manifest: manifest_path,
            out,
            model,
        } => {
            println!(
                "🏗️ Generating Candle code from manifest '{:?}'...",
                manifest_path
            );

            let manifest_content = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;

            let manifest: HashMap<String, LayerMeta> =
                serde_json::from_str(&manifest_content).context("Failed to parse manifest JSON")?;

            let generator = Codegen::new(manifest);
            let code = generator.generate_model_rs(&model);

            std::fs::write(&out, code)
                .with_context(|| format!("Failed to write generated code to {:?}", out))?;

            println!("✅ Code generated successfully at {:?}", out);
        }
    }

    Ok(())
}
