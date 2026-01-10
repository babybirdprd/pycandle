//! Neural network layer implementations for Candle
//!
//! These provide PyTorch-compatible implementations that Candle doesn't have built-in.

use candle_nn::Module;

use candle_core::{IndexOp, Result, Tensor};

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
        let alpha = vb
            .get((in_features,), "alpha")
            .or_else(|_| vb.get((1, in_features, 1), "alpha"))?;
        let alpha = alpha.reshape((1, in_features, 1))?;
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

/// Mish activation: x * tanh(softplus(x))
pub struct Mish;
impl Mish {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Mish(x) = x * tanh(softplus(x))
        // softplus(x) = ln(1 + exp(x))
        let sp = (x.exp()? + 1.0)?.log()?;
        x.broadcast_mul(&sp.tanh()?)
    }
}

/// SiLU / Swish activation: x * sigmoid(x)
pub struct SiLU;
impl SiLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x * candle_nn::ops::sigmoid(x)?
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

/// LlamaRMSNorm: Root Mean Square Layer Normalization
pub struct LlamaRMSNorm {
    pub weight: Tensor,
    pub eps: f64,
}

impl LlamaRMSNorm {
    pub fn load(vb: candle_nn::VarBuilder, size: usize, eps: f64) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = candle_core::DType::F32;
        let x = x.to_dtype(internal_dtype)?;
        let norm_x = x.sqr()?.mean_keepdim(candle_core::D::Minus1)?;
        let x_normed = x.broadcast_div(&(norm_x + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
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

// ============================================================================
// Specialized Layers
// ============================================================================

/// CausalConv1d: A 1D convolution with causal padding
/// Ensures that output at time t only depends on inputs at time <= t
pub struct CausalConv1d {
    pub conv: candle_nn::Conv1d,
    pub padding: usize,
}

impl CausalConv1d {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        bias: bool,
    ) -> Result<Self> {
        let padding = kernel_size - 1;
        let config = candle_nn::Conv1dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let conv = if bias {
            candle_nn::conv1d(in_channels, out_channels, kernel_size, config, vb)?
        } else {
            candle_nn::conv1d_no_bias(in_channels, out_channels, kernel_size, config, vb)?
        };
        Ok(Self { conv, padding })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        // Causal slice: remove the 'future' padding at the end
        if self.padding > 0 {
            let dim = x.dims().len() - 1;
            let seq_len = x.dim(dim)?;
            x.narrow(dim, 0, seq_len - self.padding)
        } else {
            Ok(x)
        }
    }
}

/// Dropout layer (inference no-op)
pub struct Dropout {
    pub p: f32,
}
impl Dropout {
    pub fn new() -> Self {
        Self { p: 0.5 }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

/// Transpose layer (swaps two dimensions)
pub struct Transpose {
    pub dim0: usize,
    pub dim1: usize,
}
impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Self {
        Self { dim0, dim1 }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.transpose(self.dim0, self.dim1)
    }
}

/// ConvTranspose1d implementation
pub struct ConvTranspose1d {
    inner: candle_nn::ConvTranspose1d,
}

impl ConvTranspose1d {
    pub fn load(
        vb: candle_nn::VarBuilder,
        in_c: usize,
        out_c: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
    ) -> Result<Self> {
        let weight = vb.get((in_c, out_c, kernel_size), "weight").or_else(|_| {
            vb.pp("parametrizations.weight")
                .get((in_c, out_c, kernel_size), "original1")
        })?;
        let bias = vb.get((out_c,), "bias").ok();

        let config = candle_nn::ConvTranspose1dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let inner = candle_nn::ConvTranspose1d::new(weight, bias, config);
        Ok(Self { inner })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.inner.forward(x)
    }
}

/// Sinusoidal Positional Embedding
pub struct SinusoidalPosEmb {
    pub dim: usize,
}
impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let device = x.device();
        let dtype = x.dtype();
        let inv_freq: Vec<_> = (0..half_dim)
            .map(|i| 1.0f32 / (10000.0f32.powf(i as f32 / (half_dim as f32 - 1.0f32))))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?.to_dtype(dtype)?;
        let emb = x.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        Tensor::cat(&[emb.sin()?, emb.cos()?], 1)
    }
}

/// Helper to load a Linear layer that uses PyTorch Weight Normalization.
/// Composes w = g * (v / ||v||) at runtime.
pub fn load_weight_norm_linear(
    vb: candle_nn::VarBuilder,
    in_f: usize,
    out_f: usize,
    bias: bool,
) -> Result<candle_nn::Linear> {
    // Attempt to load 'original1' (v) and 'original0' (g) from parametrizations
    // If not found, try legacy 'weight_v' and 'weight_g'

    let v_path = if vb.contains_tensor("parametrizations.weight.original1") {
        "parametrizations.weight.original1"
    } else {
        "weight_v"
    };

    let g_path = if vb.contains_tensor("parametrizations.weight.original0") {
        "parametrizations.weight.original0"
    } else {
        "weight_g"
    };

    let v = vb.get((out_f, in_f), v_path)?; // Shape: [out, in]
    let g = vb.get((out_f, 1), g_path)?; // Shape: [out, 1]

    // Normalize v: v / ||v||
    // Norm along dim 1 (input dimension) for Linear
    let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;

    // Compose: w = g * (v / norm)
    let w = v.broadcast_div(&v_norm)?.broadcast_mul(&g)?;

    // Transpose for Candle (Candle Linear is [out, in], but usually we transpose standard weights.
    // However, weight_norm vectors in PT are usually [out, in].
    // Candle's Linear::new expects weight to be [out, in].
    // Let's standardise on returning the layer directly.

    let b = if bias {
        Some(vb.get((out_f,), "bias")?)
    } else {
        None
    };

    Ok(candle_nn::Linear::new(w, b))
}
