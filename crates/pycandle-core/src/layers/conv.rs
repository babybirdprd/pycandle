use candle_core::{Module, Result, Tensor};

pub fn load_weight_norm_conv1d(
    vb: candle_nn::VarBuilder,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> Result<candle_nn::Conv1d> {
    let weight_g = vb
        .pp("parametrizations.weight.original0")
        .get((out_channels, 1, 1), "weight_g")?;
    let weight_v = vb
        .pp("parametrizations.weight.original1")
        .get((out_channels, in_channels, kernel_size), "weight_v")?;

    // v * (g / ||v||)
    // Norm along (1, 2) which are (in_channels, kernel_size)
    let norm_v = weight_v.sqr()?.sum_keepdim((1, 2))?.sqrt()?;
    let weight = weight_v.broadcast_mul(&weight_g.broadcast_div(&norm_v)?)?;

    let bias = vb.get(out_channels, "bias").ok();

    let config = candle_nn::Conv1dConfig {
        stride,
        padding,
        ..Default::default()
    };

    Ok(candle_nn::Conv1d::new(weight, bias, config))
}

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

/// ConvTranspose1d implementation
pub struct ConvTranspose1d {
    pub inner: candle_nn::ConvTranspose1d,
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
