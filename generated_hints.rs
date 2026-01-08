use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};

pub struct Config {
    pub hidden_dim: usize, // 64
    pub vocab_size: usize, // 64
}

pub struct MyModel {
    pub embedding: Embedding,
    pub linear: Linear,
    pub checker: Option<PyChecker>,
}

impl MyModel {
    pub fn load(cfg: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embedding"))?;
        let linear = { let w = vb.pp("linear").get((cfg.hidden_dim, cfg.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("linear").get(cfg.hidden_dim, "bias")?); Linear::new(w, b) };

        Ok(Self {
            embedding,
            linear,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: embedding
        x = self.embedding.forward(&x)?;
        py_check!(self.checker, "embedding", &x);

        // Layer: linear
        x = self.linear.forward(&x)?;
        py_check!(self.checker, "linear", &x);

        Ok(x)
    }
}
