use candle_core::{Result, Tensor};

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
