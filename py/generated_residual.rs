use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};

pub struct ResidualModel {
    pub bn1: BatchNorm1d,
    pub bn2: BatchNorm1d,
    pub conv1: Conv1d,
    pub conv2: Conv1d,
    pub relu: ReLU,
    pub relu_1: ReLU,
    pub checker: Option<PyChecker>,
}

impl ResidualModel {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let bn1 = BatchNorm1d::load(vb.pp("bn1"), 16)?;
        let bn2 = BatchNorm1d::load(vb.pp("bn2"), 16)?;
        let conv1 = candle_nn::conv1d(16, 16, 3, candle_nn::Conv1dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("conv1"))?;
        let conv2 = candle_nn::conv1d(16, 16, 3, candle_nn::Conv1dConfig { stride: 1, padding: 1, ..Default::default() }, vb.pp("conv2"))?;
        let relu = ReLU;
        let relu_1 = ReLU;

        Ok(Self {
            bn1,
            bn2,
            conv1,
            conv2,
            relu,
            relu_1,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let x_conv1 = self.conv1.forward(&xs)?;
        py_check!(self.checker, "conv1", &x_conv1);
        let x_bn1 = self.bn1.forward(&x_conv1)?;
        py_check!(self.checker, "bn1", &x_bn1);
        let x_relu = self.relu.forward(&x_bn1)?;
        py_check!(self.checker, "relu", &x_relu);
        let x_conv2 = self.conv2.forward(&x_relu)?;
        py_check!(self.checker, "conv2", &x_conv2);
        let x_bn2 = self.bn2.forward(&x_conv2)?;
        py_check!(self.checker, "bn2", &x_bn2);
        let x_add = (&x_bn2 + &xs)?;
        let x_relu_1 = self.relu.forward(&x_add)?;
        py_check!(self.checker, "relu", &x_relu_1);
        Ok(x_relu_1)
    }
}
