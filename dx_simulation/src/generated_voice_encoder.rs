use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};

pub struct MyModel {
    pub lstm: LSTM,
    pub proj: Linear,
    pub checker: Option<PyChecker>,
}

impl MyModel {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let lstm = LSTM::load(vb.pp("lstm"), 40, 256, 3)?;
        let proj = { let w = vb.pp("proj").get((256, 256), "weight")?.t()?; let b = Some(vb.pp("proj").get(256, "bias")?); Linear::new(w, b) };

        Ok(Self {
            lstm,
            proj,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: lstm
        x = self.lstm.forward(&x)?.0;
        py_check!(self.checker, "lstm", &x);

        // Layer: proj
        x = self.proj.forward(&x)?;
        py_check!(self.checker, "proj", &x);

        Ok(x)
    }
}
