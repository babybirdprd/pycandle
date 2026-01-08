use anyhow::{Context, Result};
use colored::Colorize;
use std::fs;
use std::path::Path;

const RECORDER_TEMPLATE: &str = r#"import torch
import sys
import os

# Try to import pycandle spy. 
# in a real setup this would be installed, but for now we look in relative paths common in this workspace.
try:
    from pycandle.spy import GoldenRecorder
except ImportError:
    # Add potential fallback paths
    possible_paths = ["py", "../py", "../../py"]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "spy.py")):
            sys.path.append(p)
            break
    from spy import GoldenRecorder

# TODO: Import your model class
# from my_project.model import MyModel

def main():
    print("🚀 Initializing model configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    # TODO: Instantiate your model
    # model = MyModel().to(device)
    # model.eval()

    # TODO: Create dummy input matching your model's requirement
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("🎥 Starting recording...")
    recorder = GoldenRecorder(output_dir="traces")
    
    # TODO: Run the forward pass with the recorder
    # recorder.record(model, dummy_input)
    
    # Save the trace
    name = "debug_run"
    recorder.save(name)
    print(f"✅ Recording saved to traces/{name}")

if __name__ == "__main__":
    main()
"#;

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

pub fn run_init(name: Option<String>) -> Result<()> {
    println!("{}", "⚡ PyCandle Project Initialization".bold().green());

    // 1. Create recorder.py
    let recorder_path = Path::new("recorder.py");
    if recorder_path.exists() {
        println!("   {} recorder.py already exists, skipping.", "⚠️".yellow());
    } else {
        fs::write(recorder_path, RECORDER_TEMPLATE).context("Failed to write recorder.py")?;
        println!("   {} Created recorder.py", "✅".green());
    }

    // 2. Create tests directory and parity test if requested
    let tests_dir = Path::new("tests");
    if !tests_dir.exists() {
        fs::create_dir(tests_dir).ok(); // failure ok, maybe we are not in root
    }

    let test_path = tests_dir.join("parity.rs");
    if tests_dir.exists() && !test_path.exists() {
        fs::write(&test_path, TEST_TEMPLATE).context("Failed to write tests/parity.rs")?;
        println!("   {} Created tests/parity.rs", "✅".green());
    } else {
        println!(
            "   {} tests/parity.rs already exists (or tests/ missing), skipping.",
            "⚠️".yellow()
        );
    }

    println!("\nNext steps:");
    println!("1. Edit {} to import your model.", "recorder.py".bold());
    println!(
        "2. Run {} to capture traces.",
        "pycandle record --script recorder.py --name debug_run".bold()
    );
    println!(
        "3. Run {} to generate Rust code.",
        "pycandle codegen ...".bold()
    );

    Ok(())
}
