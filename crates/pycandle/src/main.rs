//! PyCandle CLI
//!
//! Command-line interface for PyTorch → Candle porting.

mod report;
mod todos;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use pycandle_core::LayerMeta;
use pycandle_core::codegen::Codegen;
use report::ReportGenerator;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;

#[derive(Parser)]
#[command(name = "pycandle")]
#[command(about = "A tool for bit-perfect parity checking between PyTorch and Candle", long_about = None)]
struct Cli {
    /// Output in JSON format for agent consumption
    #[arg(long, global = true)]
    json: bool,

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

        /// Analyze without generating code
        #[arg(long)]
        analyze_only: bool,
    },
    /// Extract and manage TODO markers in generated code
    Todos {
        /// Path to generated Rust file
        #[arg(short, long)]
        file: PathBuf,

        /// Just check if TODOs remain (exit code 1 if any)
        #[arg(long)]
        check: bool,
    },
    /// Generate an HTML coverage report
    Report {
        /// Path to the manifest JSON file
        #[arg(short, long)]
        manifest: PathBuf,

        /// Output HTML file path
        #[arg(short, long, default_value = "pycandle_report.html")]
        out: PathBuf,
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
            analyze_only,
        } => {
            let manifest_content = std::fs::read_to_string(&manifest_path)
                .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;

            let manifest: HashMap<String, LayerMeta> =
                serde_json::from_str(&manifest_content).context("Failed to parse manifest JSON")?;

            let generator = Codegen::new(manifest);

            if analyze_only || cli.json {
                let analysis = generator.analyze();

                if cli.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&analysis)
                            .context("Failed to serialize analysis")?
                    );
                } else {
                    println!("📊 Analysis of {:?}:", manifest_path);
                    println!(
                        "  Supported: {}/{} ({:.1}%)",
                        analysis.supported, analysis.total, analysis.coverage_percent
                    );
                    println!("  Unsupported: {}", analysis.unsupported);
                    if !analysis.gaps.is_empty() {
                        println!("\n  Gaps:");
                        for gap in &analysis.gaps {
                            println!(
                                "    - {}: {} occurrence(s) → {}",
                                gap.module_type, gap.count, gap.suggestion
                            );
                        }
                    }
                }

                if !analyze_only {
                    // JSON mode but not analyze_only - still generate code
                    let code = generator.generate_model_rs(&model);
                    std::fs::write(&out, code)
                        .with_context(|| format!("Failed to write generated code to {:?}", out))?;
                }
            } else {
                println!(
                    "🏗️ Generating Candle code from manifest '{:?}'...",
                    manifest_path
                );

                let code = generator.generate_model_rs(&model);

                std::fs::write(&out, code)
                    .with_context(|| format!("Failed to write generated code to {:?}", out))?;

                println!("✅ Code generated successfully at {:?}", out);
            }
        }
        Commands::Todos { file, check } => {
            let content = std::fs::read_to_string(&file)
                .with_context(|| format!("Failed to read file at {:?}", file))?;

            let todos = todos::extract_todos(&content);
            let report = todos::generate_report(file.to_str().unwrap_or("unknown"), todos);

            if cli.json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&report).context("Failed to serialize report")?
                );
            } else {
                println!("📋 TODOs in {:?}:", file);
                println!("   Total: {}", report.total);
                if !report.by_type.is_empty() {
                    println!("\n   By type:");
                    let mut types: Vec<_> = report.by_type.iter().collect();
                    types.sort_by_key(|(_, count)| std::cmp::Reverse(*count));
                    for (t, c) in types {
                        println!("   - {}: {}", t, c);
                    }
                }
                if !report.todos.is_empty() {
                    println!("\n   Details:");
                    for todo in &report.todos {
                        println!(
                            "   L{}: {} ({}) → {}",
                            todo.line, todo.field_name, todo.module_type, todo.suggestion
                        );
                    }
                }
            }

            if check && report.total > 0 {
                std::process::exit(1);
            }
        }
        Commands::Report { manifest, out } => {
            let manifest_content = std::fs::read_to_string(&manifest)
                .with_context(|| format!("Failed to read manifest at {:?}", manifest))?;

            let manifest_data: HashMap<String, LayerMeta> =
                serde_json::from_str(&manifest_content).context("Failed to parse manifest JSON")?;

            let generator = ReportGenerator::new(manifest_data);
            let data = generator.analyze();
            let html = generator.generate_html(&data);

            std::fs::write(&out, html)
                .with_context(|| format!("Failed to write report to {:?}", out))?;

            println!("📊 Report generated: {:?}", out);
        }
    }

    Ok(())
}
