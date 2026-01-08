use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};

pub struct ComplexModel {
    pub conv1: Conv1d,
    pub fc: Linear,
    pub relu: ReLU,
    pub checker: Option<PyChecker>,
}

impl ComplexModel {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let conv1 = candle_nn::conv1d(16, 32, 3, candle_nn::Conv1dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("conv1"))?;
        let fc = candle_nn::linear(320, 10, vb.pp("fc"))?;
        let relu = ReLU;

        Ok(Self {
            conv1,
            fc,
            relu,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_conv1 = self.conv1.forward(&xs)?;
        py_check!(self.checker, "conv1", &x_conv1);
        let x_relu = self.relu.forward(&x_conv1)?;
        py_check!(self.checker, "relu", &x_relu);
        let x_fc = self.fc.forward(&view)?;
        py_check!(self.checker, "fc", &x_fc);
        Ok(x_fc)
    }
}
