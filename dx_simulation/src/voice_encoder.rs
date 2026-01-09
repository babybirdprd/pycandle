use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use pycandle_core::{PyChecker, py_check, layers::*};

pub struct Config {}
pub struct VoiceEncoder {
    pub lstm: LSTM,
    pub proj: candle_nn::Linear,
    pub checker: Option<PyChecker>,
}

impl VoiceEncoder {
    #[allow(unused_variables)]
    pub fn load(config: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let lstm = LSTM::load(vb.pp("lstm"), 40, 256, 3)?;
        let proj = {
            let w = vb.pp("proj").get((256, 256), "weight")?.t()?;
            let b = Some(vb.pp("proj").get(256, "bias")?);
            candle_nn::Linear::new(w, b)
        };
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

        // L2 normalize
        let norm = x.sqr()?.sum_keepdim(1)?.sqrt()?;
        x.broadcast_div(&norm)
    }
}
