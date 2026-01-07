//! PyChecker - Golden tensor comparison for parity verification

use candle_core::{Device, Error, Result, Shape, Tensor};
use colored::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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
pub struct ComparisonResult {
    pub mse: f32,
    pub max_diff: f32,
    pub cosine_sim: f32,
    pub passed: bool,
}

/// PyChecker loads golden tensors and verifies Rust outputs against them
pub struct PyChecker {
    pub name: String,
    golden_tensors: HashMap<String, Tensor>,
    pub manifest: HashMap<String, LayerMeta>,
    pub atol: f32,
    pub rtol: f32,
    pub device: Device,
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
        let manifest: HashMap<String, LayerMeta> = serde_json::from_str(&manifest_file)
            .map_err(|e| Error::Msg(format!("Failed to parse manifest: {}", e)))?;

        Ok(Self {
            name: project_name.to_string(),
            golden_tensors,
            manifest,
            atol: 1e-4,
            rtol: 1e-4,
            device: device.clone(),
        })
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

        let result = self.compare_tensors(actual, expected)?;

        if result.mse > self.atol {
            self.report_failure(layer_name, &result, actual, expected);
            return Err(Error::Msg(format!(
                "Numerical parity failed for {}",
                layer_name
            )));
        }

        println!(
            "{} Layer '{}' passed. (MSE: {:.2e}, CosSim: {:.4})",
            "✔".green(),
            layer_name.yellow(),
            result.mse,
            result.cosine_sim
        );
        Ok(result)
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
            mse,
            max_diff,
            cosine_sim,
            passed: mse <= self.atol,
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

        let a_mean = actual.mean_all().unwrap().to_scalar::<f32>().unwrap();
        let e_mean = expected.mean_all().unwrap().to_scalar::<f32>().unwrap();
        println!("  Means:    Rust={:.6}, Py={:.6}", a_mean, e_mean);
        println!("{}\n", "---------------------------".red());
    }
}
