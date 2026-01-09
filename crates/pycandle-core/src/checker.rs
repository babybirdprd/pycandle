//! PyChecker - Golden tensor comparison for parity verification

use candle_core::{Device, Error, Result, Shape, Tensor};
use colored::*;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

/// Mode for verification: Strict (panic on failure) or DriftTracking (record and continue)
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum VerificationMode {
    Strict,
    DriftTracking,
}

/// Metadata for a recorded layer
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerMeta {
    pub name: String,
    pub module_type: String,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
    pub parameters: Vec<String>,
    pub is_leaf: bool,
    pub config: serde_json::Value,
}

/// Result of comparing two tensors
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct ComparisonResult {
    pub name: String,
    pub mse: f32,
    pub max_diff: f32,
    pub cosine_sim: f32,
    pub passed: bool,
    pub heatmap: Option<Vec<f32>>,
}

/// PyChecker loads golden tensors and verifies Rust outputs against them
#[derive(Clone)]
pub struct PyChecker {
    pub name: String,
    golden_tensors: HashMap<String, Tensor>,
    pub manifest: HashMap<String, LayerMeta>,
    pub atol: f32,
    pub rtol: f32,
    pub device: Device,
    pub mode: VerificationMode,
    history: RefCell<Vec<ComparisonResult>>,
}

impl PyChecker {
    /// Load golden records from safetensors and manifest JSON
    pub fn load<P: AsRef<Path>>(project_name: &str, base_path: P, device: &Device) -> Result<Self> {
        let base = base_path.as_ref();
        let tensor_path = base.join(format!("{}_trace.safetensors", project_name));
        let manifest_path = base.join(format!("{}_manifest.json", project_name));

        let golden_tensors = candle_core::safetensors::load(&tensor_path, device)?;

        let manifest_file = std::fs::read_to_string(&manifest_path)
            .map_err(|e| Error::Msg(format!("Failed to read manifest: {}", e)))?;
        let full_manifest: HashMap<String, serde_json::Value> = serde_json::from_str(&manifest_file)
            .map_err(|e| Error::Msg(format!("Failed to parse manifest: {}", e)))?;

        let manifest: HashMap<String, LayerMeta> = full_manifest
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| {
                let meta: LayerMeta = serde_json::from_value(v).map_err(|e| {
                    Error::Msg(format!("Failed to parse LayerMeta for {}: {}", k, e))
                })?;
                Ok((k, meta))
            })
            .collect::<Result<HashMap<String, LayerMeta>>>()?;

        Ok(Self {
            name: project_name.to_string(),
            golden_tensors,
            manifest,
            atol: 1e-4,
            rtol: 1e-4,
            device: device.clone(),
            mode: VerificationMode::Strict,
            history: RefCell::new(Vec::new()),
        })
    }

    /// Set verification mode
    pub fn with_mode(mut self, mode: VerificationMode) -> Self {
        self.mode = mode;
        self
    }

    /// Verify a tensor against the golden record for a layer
    pub fn verify(&self, layer_name: &str, actual: &Tensor) -> Result<ComparisonResult> {
        let key = format!("{}.out.0", layer_name);
        let expected = self.golden_tensors.get(&key).ok_or_else(|| {
            Error::Msg(format!(
                "Layer '{}' not found in trace. Available: {:?}",
                layer_name,
                self.golden_tensors.keys().take(5).collect::<Vec<_>>()
            ))
        })?;

        if actual.shape() != expected.shape() {
            self.diagnose_shape_mismatch(layer_name, actual.shape(), expected.shape());
            return Err(Error::Msg(format!(
                "Shape mismatch for {}: actual {:?}, expected {:?}",
                layer_name,
                actual.shape(),
                expected.shape()
            )));
        }

        let mut result = self.compare_tensors(actual, expected)?;
        result.name = layer_name.to_string();

        if result.mse > self.atol {
            // Compute heatmap for dashboard
            result.heatmap = self.compute_heatmap(actual, expected);

            // In Strict mode, we fail immediately
            if self.mode == VerificationMode::Strict {
                self.report_failure(layer_name, &result, actual, expected);
                self.log_result(&result);
                return Err(Error::Msg(format!(
                    "Numerical parity failed for {}",
                    layer_name
                )));
            } else {
                // In DriftTracking mode, we warn but continue
                println!(
                    "{} Layer '{}' drifted. (MSE: {:.2e})",
                    "⚠".yellow(),
                    layer_name.yellow(),
                    result.mse
                );
            }
        } else {
            println!(
                "{} Layer '{}' passed. (MSE: {:.2e}, CosSim: {:.4})",
                "✔".green(),
                layer_name.yellow(),
                result.mse,
                result.cosine_sim
            );
        }

        self.log_result(&result);
        self.history.borrow_mut().push(result.clone());
        Ok(result)
    }

    /// Print a report of the most sensitive layers (highest drift)
    pub fn print_drift_report(&self) {
        let history = self.history.borrow();
        if history.is_empty() {
            return;
        }

        println!("\n{}", "📉 QUANTIZATION DRIFT REPORT".blue().bold());
        println!("{:<40} | {:<12} | {:<12}", "Layer", "MSE", "Status");
        println!("{}", "-".repeat(70));

        let mut sorted_history = history.clone();
        sorted_history.sort_by(|a, b| b.mse.partial_cmp(&a.mse).unwrap());

        for res in sorted_history.iter().take(20) {
            let status = if res.mse > self.atol {
                "DRIFT".red()
            } else {
                "OK".green()
            };
            println!("{:<40} | {:.2e}   | {}", res.name, res.mse, status);
        }
        println!("\n");
    }

    fn log_result(&self, result: &ComparisonResult) {
        if let Ok(json) = serde_json::to_string(result) {
            if let Ok(mut file) = OpenOptions::new()
                .create(true)
                .append(true)
                .open("verification_results.jsonl")
            {
                let _ = writeln!(file, "{}", json);
            }
        }
    }

    fn compare_tensors(&self, a: &Tensor, b: &Tensor) -> Result<ComparisonResult> {
        let diff = (a - b)?;
        let mse = diff.sqr()?.mean_all()?.to_scalar::<f32>()?;
        let max_diff = diff.abs()?.max_all()?.to_scalar::<f32>()?;

        let a_flat = a.flatten_all()?;
        let b_flat = b.flatten_all()?;
        let dot = (&a_flat * &b_flat)?.sum_all()?.to_scalar::<f32>()?;
        let norm_a = a_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let norm_b = b_flat.sqr()?.sum_all()?.sqrt()?.to_scalar::<f32>()?;
        let cosine_sim = dot / (norm_a * norm_b + 1e-8);

        Ok(ComparisonResult {
            name: "unknown".to_string(), // Will be overwritten by caller
            mse,
            max_diff,
            cosine_sim,
            passed: mse <= self.atol,
            heatmap: None,
        })
    }

    fn diagnose_shape_mismatch(&self, name: &str, actual: &Shape, expected: &Shape) {
        println!("\n{}", "❌ SHAPE MISMATCH DETECTED".red().bold());
        println!("Layer: {}", name.yellow());
        println!("  Rust:   {:?}", actual.dims());
        println!("  Python: {:?}", expected.dims());

        let a_dims = actual.dims();
        let e_dims = expected.dims();

        if a_dims.len() == 3 && e_dims.len() == 3 {
            if a_dims[1] == e_dims[2] && a_dims[2] == e_dims[1] {
                println!(
                    "{}",
                    "💡 DIAGNOSIS: Dimension Swap. (B, C, T) vs (B, T, C). Try .transpose(1, 2)?"
                        .cyan()
                );
            }
        }
        println!("{}\n", "---------------------------".red());
    }

    fn report_failure(
        &self,
        name: &str,
        res: &ComparisonResult,
        actual: &Tensor,
        expected: &Tensor,
    ) {
        println!("\n{}", "❌ NUMERICAL PARITY FAILED".red().bold());
        println!("Layer: {}", name.yellow());
        println!("  MSE:      {:.8}", res.mse);
        println!("  Cos Sim:  {:.8}", res.cosine_sim);

        let a_mean = actual.mean_all().unwrap().to_scalar::<f32>().unwrap_or(0.0);
        let e_mean = expected
            .mean_all()
            .unwrap()
            .to_scalar::<f32>()
            .unwrap_or(0.0);
        println!("  Means:    Rust={:.6}, Py={:.6}", a_mean, e_mean);

        // --- Active Debugging Artifacts ---
        let failures_dir = Path::new("failures");
        if !failures_dir.exists() {
            let _ = std::fs::create_dir(failures_dir);
        }

        // 1. Save Tensor Snippet (.safetensors)
        let snippet_path = failures_dir.join(format!("{}.safetensors", name));
        let tensors_to_save = HashMap::from([
            ("rust_actual".to_string(), actual.clone()),
            ("py_golden".to_string(), expected.clone()),
        ]);
        if let Err(e) = candle_core::safetensors::save(&tensors_to_save, &snippet_path) {
            println!("  Failed to save snippet: {}", e);
        } else {
            println!(
                "  💾 Snippet saved: {}",
                snippet_path.display().to_string().cyan()
            );
        }

        // 2. Generate Python Analysis Script
        let script_path = failures_dir.join(format!("debug_{}.py", name));
        let script_content = format!(
            r#"
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    print(f"🔍 Analyzing Failure: {{'{name}'}}")
    tensors = load_file("{filename}")
    rust = tensors["rust_actual"]
    gold = tensors["py_golden"]

    diff = (rust - gold).abs()
    print(f"  Max Diff: {{diff.max().item():.6f}}")
    print(f"  MSE:      {{(diff ** 2).mean().item():.8f}}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Rust Tensor Histogram")
    plt.hist(rust.flatten().float().numpy(), bins=50, alpha=0.7, label='Rust')
    plt.hist(gold.flatten().float().numpy(), bins=50, alpha=0.7, label='Gold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Difference Heatmap (First Slice)")
    if diff.ndim > 1:
        plt.imshow(diff.flatten(0, -2)[0].float().numpy(), cmap='hot', aspect='auto')
    else:
        plt.plot(diff.float().numpy())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze()
"#,
            name = name,
            filename = format!("{}.safetensors", name)
        );

        if let Ok(mut file) = std::fs::File::create(&script_path) {
            let _ = file.write_all(script_content.as_bytes());
            println!(
                "  🐍 Script generated: {}",
                script_path.display().to_string().cyan()
            );
        }

        println!("{}\n", "---------------------------".red());
    }

    pub fn compute_heatmap(&self, a: &Tensor, b: &Tensor) -> Option<Vec<f32>> {
        // Compute absolute difference
        let diff = (a - b).ok()?.abs().ok()?;

        // 1. Flatten to 1D
        let flat = diff.flatten_all().ok()?;
        let numel = flat.elem_count();

        // 2. We want an 8x8 grid = 64 buckets
        let grid_size = 64;
        if numel < grid_size {
            return None; // Too small to pool usefuly
        }

        let chunk_size = numel / grid_size;

        // 3. Simple max pooling into buckets
        // Walking through the tensor on CPU is slow for huge tensors, but this is a failure case anyway.
        // A faster way is to reshape (64, chunk_size) and max(dim=1).

        let reshaped = flat
            .narrow(0, 0, grid_size * chunk_size)
            .ok()?
            .reshape((grid_size, chunk_size))
            .ok()?;

        let pooled = reshaped.max(1).ok()?;

        // Convert to Vec<f32>
        pooled.to_vec1::<f32>().ok()
    }
}
