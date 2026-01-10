use candle_core::{Result, Tensor};

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
