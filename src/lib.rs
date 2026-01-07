use candle_core::{Device, Error, IndexOp, Result, Shape, Tensor};
use colored::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

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

pub struct ComparisonResult {
    pub mse: f32,
    pub max_diff: f32,
    pub cosine_sim: f32,
    pub passed: bool,
}

pub struct PyChecker {
    pub name: String,
    golden_tensors: HashMap<String, Tensor>,
    pub manifest: HashMap<String, LayerMeta>,
    pub atol: f32,
    pub rtol: f32,
    pub device: Device,
}

impl PyChecker {
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

#[macro_export]
macro_rules! py_check {
    ($checker:expr, $name:expr, $tensor:expr) => {
        if let Some(ref c) = $checker {
            c.verify($name, $tensor).expect("Parity Check Failed");
        }
    };
}

// ============================================================================
// Activation Functions
// ============================================================================

/// ReLU activation: max(0, x)
pub struct ReLU;
impl ReLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.relu()
    }
}

/// GELU activation (Gaussian Error Linear Unit)
pub struct GELU;
impl GELU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.gelu_erf()
    }
}

/// Sigmoid activation: 1 / (1 + exp(-x))
pub struct Sigmoid;
impl Sigmoid {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(x)
    }
}

/// Tanh activation
pub struct Tanh;
impl Tanh {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.tanh()
    }
}

/// ELU activation: x if x > 0, else alpha * (exp(x) - 1)
pub struct ELU {
    pub alpha: f64,
}
impl ELU {
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.elu(self.alpha)
    }
}

/// LeakyReLU activation: x if x > 0, else negative_slope * x
pub struct LeakyReLU {
    pub negative_slope: f64,
}
impl LeakyReLU {
    pub fn new(negative_slope: f64) -> Self {
        Self { negative_slope }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // leaky_relu: max(0, x) + negative_slope * min(0, x)
        let zeros = x.zeros_like()?;
        let pos = x.maximum(&zeros)?;
        let neg = x.minimum(&zeros)?;
        pos + (neg * self.negative_slope)?
    }
}

/// Snake activation: x + sin²(αx)/α
/// Used in neural vocoders like BigVGAN
pub struct Snake {
    pub alpha: Tensor,
}
impl Snake {
    pub fn load(vb: candle_nn::VarBuilder, in_features: usize) -> Result<Self> {
        let alpha = vb.get((1, in_features, 1), "alpha")?;
        Ok(Self { alpha })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T), alpha: (1, C, 1)
        let ax = x.broadcast_mul(&self.alpha)?;
        let sin_ax = ax.sin()?;
        let sin_sq = sin_ax.sqr()?;
        x + sin_sq.broadcast_div(&self.alpha)?
    }
}

// ============================================================================
// Normalization Layers
// ============================================================================

/// BatchNorm1d for inference (uses running statistics)
/// Input: (B, C, T) or (B, C)
pub struct BatchNorm1d {
    pub weight: Tensor, // gamma
    pub bias: Tensor,   // beta
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
}

impl BatchNorm1d {
    pub fn load(vb: candle_nn::VarBuilder, num_features: usize) -> Result<Self> {
        let weight = vb.get((num_features,), "weight")?;
        let bias = vb.get((num_features,), "bias")?;
        let running_mean = vb.get((num_features,), "running_mean")?;
        let running_var = vb.get((num_features,), "running_var")?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T) for 1d or (B, C)
        // Normalize: (x - mean) / sqrt(var + eps) * weight + bias
        let ndim = x.dims().len();
        let (mean, var, weight, bias) = if ndim == 3 {
            // (B, C, T) - unsqueeze to (1, C, 1)
            (
                self.running_mean.unsqueeze(0)?.unsqueeze(2)?,
                self.running_var.unsqueeze(0)?.unsqueeze(2)?,
                self.weight.unsqueeze(0)?.unsqueeze(2)?,
                self.bias.unsqueeze(0)?.unsqueeze(2)?,
            )
        } else {
            // (B, C) - unsqueeze to (1, C)
            (
                self.running_mean.unsqueeze(0)?,
                self.running_var.unsqueeze(0)?,
                self.weight.unsqueeze(0)?,
                self.bias.unsqueeze(0)?,
            )
        };

        let normalized = x
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normalized.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}

/// BatchNorm2d for inference (uses running statistics)
/// Input: (B, C, H, W)
pub struct BatchNorm2d {
    pub weight: Tensor,
    pub bias: Tensor,
    pub running_mean: Tensor,
    pub running_var: Tensor,
    pub eps: f64,
}

impl BatchNorm2d {
    pub fn load(vb: candle_nn::VarBuilder, num_features: usize) -> Result<Self> {
        let weight = vb.get((num_features,), "weight")?;
        let bias = vb.get((num_features,), "bias")?;
        let running_mean = vb.get((num_features,), "running_mean")?;
        let running_var = vb.get((num_features,), "running_var")?;
        Ok(Self {
            weight,
            bias,
            running_mean,
            running_var,
            eps: 1e-5,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, H, W)
        // Reshape stats to (1, C, 1, 1) for broadcasting
        let mean = self.running_mean.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let var = self.running_var.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let weight = self.weight.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;
        let bias = self.bias.unsqueeze(0)?.unsqueeze(2)?.unsqueeze(3)?;

        let normalized = x
            .broadcast_sub(&mean)?
            .broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normalized.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}

// ============================================================================
// Recurrent Layers
// ============================================================================

/// LSTM layer (multi-layer, unidirectional)
/// Input: (B, T, input_size) if batch_first=true
/// Output: (output, (h_n, c_n))
pub struct LSTM {
    pub weight_ih: Vec<Tensor>, // One per layer: (4*hidden, input_size or hidden_size)
    pub weight_hh: Vec<Tensor>, // One per layer: (4*hidden, hidden_size)
    pub bias_ih: Vec<Tensor>,   // One per layer: (4*hidden,)
    pub bias_hh: Vec<Tensor>,   // One per layer: (4*hidden,)
    pub num_layers: usize,
    pub hidden_size: usize,
}

impl LSTM {
    pub fn load(
        vb: candle_nn::VarBuilder,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            weight_ih.push(vb.get((4 * hidden_size, in_size), &format!("weight_ih_l{}", layer))?);
            weight_hh.push(vb.get(
                (4 * hidden_size, hidden_size),
                &format!("weight_hh_l{}", layer),
            )?);
            bias_ih.push(vb.get((4 * hidden_size,), &format!("bias_ih_l{}", layer))?);
            bias_hh.push(vb.get((4 * hidden_size,), &format!("bias_hh_l{}", layer))?);
        }

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            num_layers,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, (Tensor, Tensor))> {
        // x: (B, T, input_size) assuming batch_first
        let (batch, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let h = Tensor::zeros((self.num_layers, batch, self.hidden_size), dtype, device)?;
        let c = Tensor::zeros((self.num_layers, batch, self.hidden_size), dtype, device)?;
        let mut output = x.clone();

        for layer in 0..self.num_layers {
            let mut h_t = h.i(layer)?;
            let mut c_t = c.i(layer)?;
            let mut outputs = Vec::new();

            for t in 0..seq_len {
                let x_t = output.i((.., t, ..))?;

                // gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
                let gates = x_t
                    .matmul(&self.weight_ih[layer].t()?)?
                    .broadcast_add(&h_t.matmul(&self.weight_hh[layer].t()?)?)?
                    .broadcast_add(&self.bias_ih[layer])?
                    .broadcast_add(&self.bias_hh[layer])?;

                // Split into i, f, g, o (each of size hidden_size)
                let chunks = gates.chunk(4, 1)?;
                let i_gate = candle_nn::ops::sigmoid(&chunks[0])?;
                let f_gate = candle_nn::ops::sigmoid(&chunks[1])?;
                let g_gate = chunks[2].tanh()?;
                let o_gate = candle_nn::ops::sigmoid(&chunks[3])?;

                c_t = f_gate
                    .broadcast_mul(&c_t)?
                    .broadcast_add(&i_gate.broadcast_mul(&g_gate)?)?;
                h_t = o_gate.broadcast_mul(&c_t.tanh()?)?;

                outputs.push(h_t.unsqueeze(1)?);
            }

            output = Tensor::cat(&outputs, 1)?;
        }

        Ok((output, (h, c)))
    }
}
