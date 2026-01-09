//! PyCandle CLI
//!
//! Command-line interface for PyTorch → Candle porting.

mod dashboard;
mod init;
mod report;
mod test_gen;
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

        /// Output path for the generated Rust file or directory
        #[arg(short, long)]
        out: PathBuf,

        /// Name of the model struct to generate
        #[arg(long, default_value = "MyModel")]
        model: String,

        /// Analyze without generating code
        #[arg(long)]
        analyze_only: bool,

        /// Generate stateful code with KV-caching support
        #[arg(long)]
        stateful: bool,
    },
    /// Extract and manage TODO markers in generated code
    Todos {
        /// Path to generated Rust file or directory
        #[arg(short, long)]
        path: PathBuf,

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
    Weights {
        #[command(subcommand)]
        action: WeightActions,
    },
    /// Launch the TUI Parity Dashboard
    Dashboard {
        // Optional arguments if we want to pass filter to cargo test
        #[arg(last = true)]
        args: Vec<String>,
    },
    /// Initialize a new project with boilerplate
    Init {
        /// Optional project name
        #[arg(short, long)]
        name: Option<String>,
    },
    /// Generate automated parity test
    GenTest {
        /// Name of the model struct (must match generated code)
        #[arg(long, default_value = "MyModel")]
        model: String,

        /// Path to the manifest JSON file
        #[arg(short, long)]
        manifest: PathBuf,

        /// Output path for the generated test file
        #[arg(short, long, default_value = "tests/parity.rs")]
        out: PathBuf,
    },
    /// Convert ONNX model to PyCandle manifest
    OnnxConvert {
        /// Path to the ONNX model file
        #[arg(short = 'i', long)]
        onnx: PathBuf,

        /// Project name
        #[arg(short, long)]
        name: String,

        /// Output directory for the manifest
        #[arg(short, long, default_value = "pycandle_trace")]
        out: PathBuf,
    },
}

#[derive(Subcommand)]
enum WeightActions {
    /// Surgically extract weights used in a manifest
    Extract {
        /// Path to PyTorch checkpoint (.bin, .pt, .safetensors)
        #[arg(short, long)]
        checkpoint: PathBuf,

        /// Path to the manifest JSON file
        #[arg(short, long)]
        manifest: PathBuf,

        /// Output .safetensors path
        #[arg(short, long)]
        out: PathBuf,

        /// Optional JSON mapping file for renaming
        #[arg(long)]
        map: Option<PathBuf>,
    },
    /// Rename keys in a safetensors file using a mapping
    Map {
        /// Input .safetensors file
        #[arg(short, long)]
        input: PathBuf,

        /// Output .safetensors file
        #[arg(short, long)]
        out: PathBuf,

        /// JSON mapping file
        #[arg(short, long)]
        map: PathBuf,
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
            manifest: start_path,
            out: out_path,
            model,
            analyze_only,
            stateful,
        } => {
            // Find manifests: if directory, glob *.json, else use file
            let manifest_files = if start_path.is_dir() {
                std::fs::read_dir(&start_path)?
                    .filter_map(|entry| {
                        let path = entry.ok()?.path();
                        if path.extension()?.to_str()? == "json"
                            && path.file_name()?.to_str()?.ends_with("_manifest.json")
                        {
                            Some(path)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![start_path]
            };

            if manifest_files.is_empty() {
                eprintln!("❌ No manifest files found.");
                return Ok(());
            }

            for manifest_path in manifest_files {
                let manifest_content = std::fs::read_to_string(&manifest_path)
                    .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;

                // Full manifest structure including optional graph
                #[derive(serde::Deserialize)]
                struct Manifest {
                    #[serde(flatten)]
                    layers: HashMap<String, serde_json::Value>,
                    #[serde(rename = "_graph_nodes")]
                    graph_nodes: Option<Vec<pycandle_core::codegen::GraphNode>>,
                    #[serde(rename = "_graph_code")]
                    graph_code: Option<String>,
                    #[serde(rename = "_symbolic_hints")]
                    symbolic_hints: Option<HashMap<String, usize>>,
                }

                let full_manifest: Manifest = serde_json::from_str(&manifest_content)
                    .context("Failed to parse manifest JSON")?;

                // Filter out internal keys starting with "_"
                let layers: HashMap<String, LayerMeta> = full_manifest
                    .layers
                    .into_iter()
                    .filter(|(k, _)| !k.starts_with('_'))
                    .map(|(k, v)| {
                        let meta: LayerMeta = serde_json::from_value(v)
                            .with_context(|| format!("Failed to parse LayerMeta for {}", k))?;
                        Ok((k, meta))
                    })
                    .collect::<Result<_>>()?;

                let mut generator =
                    Codegen::new(layers, full_manifest.symbolic_hints).with_stateful(stateful);
                if let Some(nodes) = full_manifest.graph_nodes {
                    generator = generator.with_graph(nodes);
                }

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
                }

                if !analyze_only {
                    // Determine output path
                    let final_out = if out_path.is_dir() {
                        let stem = manifest_path.file_stem().unwrap().to_str().unwrap();
                        let name = stem.replace("_manifest", "");
                        out_path.join(format!("generated_{}.rs", name))
                    } else {
                        // If multiple manifests but one output file, this is ambiguous/wrong unless overwriting.
                        // We'll enforce directory output if input is directory.
                        out_path.clone()
                    };

                    println!(
                        "🏗️ Generating Candle code from manifest '{:?}'...",
                        manifest_path
                    );

                    // Use model name from CLI args for struct name.
                    // TODO: maybe derive struct name from manifest filename too?
                    let code = generator.generate_model_rs(&model);

                    std::fs::write(&final_out, code).with_context(|| {
                        format!("Failed to write generated code to {:?}", final_out)
                    })?;

                    println!("✅ Code generated successfully at {:?}", final_out);
                }
            }
        }
        Commands::Todos { path, check } => {
            let files = if path.is_dir() {
                // simple recursion or flat for now? Let's just do one level or walkdir if needed.
                // For now, let's use fs::read_dir and filter .rs
                std::fs::read_dir(&path)?
                    .filter_map(|entry| {
                        let p = entry.ok()?.path();
                        if p.extension()?.to_str()? == "rs" {
                            Some(p)
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
            } else {
                vec![path]
            };

            let mut any_todos = false;

            for file_path in files {
                let content = std::fs::read_to_string(&file_path)
                    .with_context(|| format!("Failed to read file at {:?}", file_path))?;

                let todos = todos::extract_todos(&content);
                if !todos.is_empty() {
                    any_todos = true;
                }

                let report = todos::generate_report(file_path.to_str().unwrap_or("unknown"), todos);

                if cli.json {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&report)
                            .context("Failed to serialize report")?
                    );
                } else {
                    println!("📋 TODOs in {:?}:", file_path);
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
            }

            if check && any_todos {
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
        Commands::Weights { action } => match action {
            WeightActions::Extract {
                checkpoint,
                manifest,
                out,
                map,
            } => {
                println!("🔪 Performing surgical weight extraction...");
                let mut cmd = Command::new("uv");
                cmd.arg("run")
                    .arg("python")
                    .arg("py/weight_extractor.py")
                    .arg("--checkpoint")
                    .arg(&checkpoint)
                    .arg("--manifest")
                    .arg(&manifest)
                    .arg("--out")
                    .arg(&out);

                if let Some(m) = map {
                    cmd.arg("--map").arg(m);
                }

                let status = cmd.spawn()?.wait()?;
                if status.success() {
                    println!("✅ Extraction complete: {:?}", out);
                } else {
                    anyhow::bail!("Weight extraction failed");
                }
            }
            WeightActions::Map { input, out, map } => {
                println!("🔄 Renaming weights using map {:?}...", map);
                let map_content = std::fs::read_to_string(&map)?;
                let mapper = pycandle_core::WeightMapper::from_json(&map_content)?;

                let weights = candle_core::safetensors::load(&input, &candle_core::Device::Cpu)?;
                let mut renamed_weights = HashMap::new();
                for (name, tensor) in weights {
                    renamed_weights.insert(mapper.map_key(&name), tensor);
                }

                candle_core::safetensors::save(&renamed_weights, &out)?;
                println!("✅ Renaming complete: {:?}", out);
            }
        },
        Commands::Dashboard { args } => {
            dashboard::run_dashboard(&args)?;
        }
        Commands::Init { name } => {
            init::run_init(name)?;
        }
        Commands::GenTest {
            model,
            manifest,
            out,
        } => {
            println!("🧪 Generating test harness for model '{}'...", model);
            let generator = test_gen::TestGenerator::new(model, manifest)?;
            let code = generator.generate_test_file();

            if let Some(parent) = out.parent() {
                std::fs::create_dir_all(parent)
                    .with_context(|| format!("Failed to create directory {:?}", parent))?;
            }

            std::fs::write(&out, code)
                .with_context(|| format!("Failed to write test file to {:?}", out))?;

            println!("✅ Test generated at {:?}", out);
        }
        Commands::OnnxConvert { onnx, name, out } => {
            println!(
                "📦 Converting ONNX model '{:?}' to PyCandle manifest...",
                onnx
            );

            let status = Command::new("uv")
                .arg("run")
                .arg("--project")
                .arg("py")
                .env("PYTHONPATH", "py")
                .arg("python")
                .arg("py/onnx_to_fx.py")
                .arg("--onnx")
                .arg(&onnx)
                .arg("--name")
                .arg(&name)
                .arg("--out")
                .arg(&out)
                .spawn()
                .context("Failed to spawn uv run")?
                .wait()
                .context("Failed to wait for python process")?;

            if status.success() {
                println!("✅ Conversion complete. Manifest saved in {:?}", out);
            } else {
                eprintln!("❌ Conversion failed.");
            }
        }
    }

    Ok(())
}
