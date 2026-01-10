use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::Path;

const TEST_TEMPLATE: &str = r#"#[cfg(test)]
mod tests {
    use anyhow::Result;
    // use super::*; // Import your model

    #[test]
    fn test_parity() -> Result<()> {
        // TODO: Load the checker
        // let device = candle_core::Device::Cpu;
        // let checker = pycandle_core::PyChecker::load("debug_run", "traces/", &device)?;
        
        // TODO: Load your model and run forward pass
        // let model = MyModel::load(..., Some(checker))?;
        // let output = model.forward(&input)?;

        Ok(())
    }
}
"#;

pub fn run_init(_name: Option<String>) -> Result<()> {
    println!("{}", "⚡ PyCandle Project Initialization".bold().green());

    let root = Path::new(".pycandle");
    let scripts_dir = root.join("scripts");
    let generated_dir = root.join("generated");
    let venv_dir = root.join("venv");

    // 1. Create Directory Structure
    // .pycandle/
    // ├── scripts/
    // ├── generated/
    // └── venv/ (managed by uv)
    if !root.exists() {
        fs::create_dir(root).context("Failed to create .pycandle directory")?;
        println!("   {} Created .pycandle/", "✅".green());
    }

    if !scripts_dir.exists() {
        fs::create_dir(&scripts_dir).context("Failed to create .pycandle/scripts directory")?;
        println!("   {} Created .pycandle/scripts/", "✅".green());
    }

    if !generated_dir.exists() {
        fs::create_dir(&generated_dir).context("Failed to create .pycandle/generated directory")?;
        println!("   {} Created .pycandle/generated/", "✅".green());
    }

    // 2. Write Embedded Python Assets
    let assets = [
        ("spy.py", crate::python_assets::SPY_PY),
        ("recorder.py", crate::python_assets::RECORDER_TEMPLATE_PY),
        (
            "weight_extractor.py",
            crate::python_assets::WEIGHT_EXTRACTOR_PY,
        ),
        ("onnx_to_fx.py", crate::python_assets::ONNX_TO_FX_PY),
    ];

    for (filename, content) in assets {
        let path = scripts_dir.join(filename);
        if !path.exists() {
            fs::write(&path, content).with_context(|| format!("Failed to write {}", filename))?;
            println!("   {} Wrote .pycandle/scripts/{}", "✅".green(), filename);
        } else {
            println!(
                "   {} .pycandle/scripts/{} already exists, skipping.",
                "⚠️".yellow(),
                filename
            );
        }
    }

    // 3. Setup Python Environment with uv
    // Check if uv is installed
    let uv_check = std::process::Command::new("uv").arg("--version").output();

    match uv_check {
        Ok(_) => {
            println!(
                "   {} Found uv, setting up virtual environment...",
                "✅".green()
            );

            // uv venv .pycandle/venv
            if !venv_dir.exists() {
                let venv_status = std::process::Command::new("uv")
                    .arg("venv")
                    .arg(&venv_dir)
                    .status()
                    .context("Failed to run 'uv venv'")?;

                if venv_status.success() {
                    println!(
                        "   {} Created virtual environment at .pycandle/venv",
                        "✅".green()
                    );
                } else {
                    println!("   {} 'uv venv' failed.", "❌".red());
                }
            } else {
                println!(
                    "   {} .pycandle/venv already exists, skipping creation.",
                    "⚠️".yellow()
                );
            }

            // Install clear dependencies
            // We use 'uv pip install' into that venv
            println!("   ⏳ Installing dependencies (torch, safetensors, transformers, onnx)...");

            // Determine python path (windows vs unix)
            // #[cfg(windows)]
            // let python_path = venv_dir.join("Scripts").join("python.exe");
            // #[cfg(not(windows))]
            // let python_path = venv_dir.join("bin").join("python");

            // We can use `uv pip install` directly by pointing to the environment if we encourage that,
            // or just use the python binary to call pip? uv recommends `uv pip install --python <venv> ...`
            let install_status = std::process::Command::new("uv")
                .arg("pip")
                .arg("install")
                .arg("--python")
                .arg(&venv_dir) // implicit python path resolution by uv
                .arg("torch")
                .arg("safetensors")
                .arg("transformers") // usually needed for HF models
                .arg("accelerate") // good for large model loading
                .arg("onnx")
                .arg("onnx2torch")
                .status()
                .context("Failed to install dependencies");

            match install_status {
                Ok(status) if status.success() => {
                    println!("   {} Dependencies installed successfully.", "✅".green());
                }
                _ => {
                    println!(
                        "   {} Failed to install dependencies. You may need to run 'uv pip install torch safetensors transformers onnx onnx2torch' manually.",
                        "❌".red()
                    );
                }
            }
        }
        Err(_) => {
            println!(
                "   {} uv not found. Please install uv (https://github.com/astral-sh/uv) to manage dependencies automatically.",
                "⚠️".yellow()
            );
            println!(
                "   You will need to create a venv and install: torch, safetensors, transformers manually."
            );
        }
    }

    // 4. Create tests directory (Legacy/Standard)
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        fs::create_dir(tests_dir).ok();
    }
    let test_path = tests_dir.join("parity.rs");
    if tests_dir.exists() && !test_path.exists() {
        fs::write(&test_path, TEST_TEMPLATE).context("Failed to write tests/parity.rs")?;
        println!("   {} Created tests/parity.rs", "✅".green());
    }

    println!("\nNext steps:");
    println!(
        "1. Edit {} to import your model.",
        ".pycandle/scripts/recorder.py".bold()
    );
    println!("2. Run {} to capture traces.", "pycandle record".bold());

    Ok(())
}
