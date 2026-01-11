use candle_core::{Result, Tensor};

/// BatchNorm1d for inference (uses running statistics)
/// Input: (B, C, T) or (B, C)
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
#[derive(Clone, Debug)]
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
