use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};

pub struct Config {
    pub hidden_dim: usize, // 768
    pub vocab_size: usize, // 50257
}

pub struct MyTestModel {
    pub embedding: Embedding,
    pub linear1: Linear,
    pub linear2: Linear,
    pub ln: LayerNorm,
    pub checker: Option<PyChecker>,
}

impl MyTestModel {
    pub fn load(cfg: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let embedding = candle_nn::embedding(cfg.vocab_size, cfg.hidden_dim, vb.pp("embedding"))?;
        let linear1 = candle_nn::linear(cfg.hidden_dim, 2048, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(2048, cfg.hidden_dim, vb.pp("linear2"))?;
        let ln = candle_nn::layer_norm(768, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("ln"))?;

        Ok(Self {
            embedding,
            linear1,
            linear2,
            ln,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: embedding
        x = self.embedding.forward(&x)?;
        py_check!(self.checker, "embedding", &x);

        // Layer: linear1
        x = self.linear1.forward(&x)?;
        py_check!(self.checker, "linear1", &x);

        // Layer: linear2
        x = self.linear2.forward(&x)?;
        py_check!(self.checker, "linear2", &x);

        // Layer: ln
        x = self.ln.forward(&x)?;
        py_check!(self.checker, "ln", &x);

        Ok(x)
    }
}
