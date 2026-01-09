use candle_core::{Tensor, Result, Device, Shape};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};
use pycandle_core::gpt2;

pub struct MyModel {
    pub drop: Dropout,
    pub h_0_attn_c_attn: Linear,
    pub h_0_attn_c_proj: Linear,
    pub h_0_attn_resid_dropout: Dropout,
    pub h_0_ln_1: LayerNorm,
    pub h_0_ln_2: LayerNorm,
    pub h_0_mlp_act: candle_nn::Activation,
    pub h_0_mlp_c_fc: Linear,
    pub h_0_mlp_c_proj: Linear,
    pub h_0_mlp_dropout: Dropout,
    pub h_1_attn_c_attn: Linear,
    pub h_1_attn_c_proj: Linear,
    pub h_1_attn_resid_dropout: Dropout,
    pub h_1_ln_1: LayerNorm,
    pub h_1_ln_2: LayerNorm,
    pub h_1_mlp_act: candle_nn::Activation,
    pub h_1_mlp_c_fc: Linear,
    pub h_1_mlp_c_proj: Linear,
    pub h_1_mlp_dropout: Dropout,
    pub h_10_attn_c_attn: Linear,
    pub h_10_attn_c_proj: Linear,
    pub h_10_attn_resid_dropout: Dropout,
    pub h_10_ln_1: LayerNorm,
    pub h_10_ln_2: LayerNorm,
    pub h_10_mlp_act: candle_nn::Activation,
    pub h_10_mlp_c_fc: Linear,
    pub h_10_mlp_c_proj: Linear,
    pub h_10_mlp_dropout: Dropout,
    pub h_11_attn_c_attn: Linear,
    pub h_11_attn_c_proj: Linear,
    pub h_11_attn_resid_dropout: Dropout,
    pub h_11_ln_1: LayerNorm,
    pub h_11_ln_2: LayerNorm,
    pub h_11_mlp_act: candle_nn::Activation,
    pub h_11_mlp_c_fc: Linear,
    pub h_11_mlp_c_proj: Linear,
    pub h_11_mlp_dropout: Dropout,
    pub h_12_attn_c_attn: Linear,
    pub h_12_attn_c_proj: Linear,
    pub h_12_attn_resid_dropout: Dropout,
    pub h_12_ln_1: LayerNorm,
    pub h_12_ln_2: LayerNorm,
    pub h_12_mlp_act: candle_nn::Activation,
    pub h_12_mlp_c_fc: Linear,
    pub h_12_mlp_c_proj: Linear,
    pub h_12_mlp_dropout: Dropout,
    pub h_13_attn_c_attn: Linear,
    pub h_13_attn_c_proj: Linear,
    pub h_13_attn_resid_dropout: Dropout,
    pub h_13_ln_1: LayerNorm,
    pub h_13_ln_2: LayerNorm,
    pub h_13_mlp_act: candle_nn::Activation,
    pub h_13_mlp_c_fc: Linear,
    pub h_13_mlp_c_proj: Linear,
    pub h_13_mlp_dropout: Dropout,
    pub h_14_attn_c_attn: Linear,
    pub h_14_attn_c_proj: Linear,
    pub h_14_attn_resid_dropout: Dropout,
    pub h_14_ln_1: LayerNorm,
    pub h_14_ln_2: LayerNorm,
    pub h_14_mlp_act: candle_nn::Activation,
    pub h_14_mlp_c_fc: Linear,
    pub h_14_mlp_c_proj: Linear,
    pub h_14_mlp_dropout: Dropout,
    pub h_15_attn_c_attn: Linear,
    pub h_15_attn_c_proj: Linear,
    pub h_15_attn_resid_dropout: Dropout,
    pub h_15_ln_1: LayerNorm,
    pub h_15_ln_2: LayerNorm,
    pub h_15_mlp_act: candle_nn::Activation,
    pub h_15_mlp_c_fc: Linear,
    pub h_15_mlp_c_proj: Linear,
    pub h_15_mlp_dropout: Dropout,
    pub h_16_attn_c_attn: Linear,
    pub h_16_attn_c_proj: Linear,
    pub h_16_attn_resid_dropout: Dropout,
    pub h_16_ln_1: LayerNorm,
    pub h_16_ln_2: LayerNorm,
    pub h_16_mlp_act: candle_nn::Activation,
    pub h_16_mlp_c_fc: Linear,
    pub h_16_mlp_c_proj: Linear,
    pub h_16_mlp_dropout: Dropout,
    pub h_17_attn_c_attn: Linear,
    pub h_17_attn_c_proj: Linear,
    pub h_17_attn_resid_dropout: Dropout,
    pub h_17_ln_1: LayerNorm,
    pub h_17_ln_2: LayerNorm,
    pub h_17_mlp_act: candle_nn::Activation,
    pub h_17_mlp_c_fc: Linear,
    pub h_17_mlp_c_proj: Linear,
    pub h_17_mlp_dropout: Dropout,
    pub h_18_attn_c_attn: Linear,
    pub h_18_attn_c_proj: Linear,
    pub h_18_attn_resid_dropout: Dropout,
    pub h_18_ln_1: LayerNorm,
    pub h_18_ln_2: LayerNorm,
    pub h_18_mlp_act: candle_nn::Activation,
    pub h_18_mlp_c_fc: Linear,
    pub h_18_mlp_c_proj: Linear,
    pub h_18_mlp_dropout: Dropout,
    pub h_19_attn_c_attn: Linear,
    pub h_19_attn_c_proj: Linear,
    pub h_19_attn_resid_dropout: Dropout,
    pub h_19_ln_1: LayerNorm,
    pub h_19_ln_2: LayerNorm,
    pub h_19_mlp_act: candle_nn::Activation,
    pub h_19_mlp_c_fc: Linear,
    pub h_19_mlp_c_proj: Linear,
    pub h_19_mlp_dropout: Dropout,
    pub h_2_attn_c_attn: Linear,
    pub h_2_attn_c_proj: Linear,
    pub h_2_attn_resid_dropout: Dropout,
    pub h_2_ln_1: LayerNorm,
    pub h_2_ln_2: LayerNorm,
    pub h_2_mlp_act: candle_nn::Activation,
    pub h_2_mlp_c_fc: Linear,
    pub h_2_mlp_c_proj: Linear,
    pub h_2_mlp_dropout: Dropout,
    pub h_20_attn_c_attn: Linear,
    pub h_20_attn_c_proj: Linear,
    pub h_20_attn_resid_dropout: Dropout,
    pub h_20_ln_1: LayerNorm,
    pub h_20_ln_2: LayerNorm,
    pub h_20_mlp_act: candle_nn::Activation,
    pub h_20_mlp_c_fc: Linear,
    pub h_20_mlp_c_proj: Linear,
    pub h_20_mlp_dropout: Dropout,
    pub h_21_attn_c_attn: Linear,
    pub h_21_attn_c_proj: Linear,
    pub h_21_attn_resid_dropout: Dropout,
    pub h_21_ln_1: LayerNorm,
    pub h_21_ln_2: LayerNorm,
    pub h_21_mlp_act: candle_nn::Activation,
    pub h_21_mlp_c_fc: Linear,
    pub h_21_mlp_c_proj: Linear,
    pub h_21_mlp_dropout: Dropout,
    pub h_22_attn_c_attn: Linear,
    pub h_22_attn_c_proj: Linear,
    pub h_22_attn_resid_dropout: Dropout,
    pub h_22_ln_1: LayerNorm,
    pub h_22_ln_2: LayerNorm,
    pub h_22_mlp_act: candle_nn::Activation,
    pub h_22_mlp_c_fc: Linear,
    pub h_22_mlp_c_proj: Linear,
    pub h_22_mlp_dropout: Dropout,
    pub h_23_attn_c_attn: Linear,
    pub h_23_attn_c_proj: Linear,
    pub h_23_attn_resid_dropout: Dropout,
    pub h_23_ln_1: LayerNorm,
    pub h_23_ln_2: LayerNorm,
    pub h_23_mlp_act: candle_nn::Activation,
    pub h_23_mlp_c_fc: Linear,
    pub h_23_mlp_c_proj: Linear,
    pub h_23_mlp_dropout: Dropout,
    pub h_3_attn_c_attn: Linear,
    pub h_3_attn_c_proj: Linear,
    pub h_3_attn_resid_dropout: Dropout,
    pub h_3_ln_1: LayerNorm,
    pub h_3_ln_2: LayerNorm,
    pub h_3_mlp_act: candle_nn::Activation,
    pub h_3_mlp_c_fc: Linear,
    pub h_3_mlp_c_proj: Linear,
    pub h_3_mlp_dropout: Dropout,
    pub h_4_attn_c_attn: Linear,
    pub h_4_attn_c_proj: Linear,
    pub h_4_attn_resid_dropout: Dropout,
    pub h_4_ln_1: LayerNorm,
    pub h_4_ln_2: LayerNorm,
    pub h_4_mlp_act: candle_nn::Activation,
    pub h_4_mlp_c_fc: Linear,
    pub h_4_mlp_c_proj: Linear,
    pub h_4_mlp_dropout: Dropout,
    pub h_5_attn_c_attn: Linear,
    pub h_5_attn_c_proj: Linear,
    pub h_5_attn_resid_dropout: Dropout,
    pub h_5_ln_1: LayerNorm,
    pub h_5_ln_2: LayerNorm,
    pub h_5_mlp_act: candle_nn::Activation,
    pub h_5_mlp_c_fc: Linear,
    pub h_5_mlp_c_proj: Linear,
    pub h_5_mlp_dropout: Dropout,
    pub h_6_attn_c_attn: Linear,
    pub h_6_attn_c_proj: Linear,
    pub h_6_attn_resid_dropout: Dropout,
    pub h_6_ln_1: LayerNorm,
    pub h_6_ln_2: LayerNorm,
    pub h_6_mlp_act: candle_nn::Activation,
    pub h_6_mlp_c_fc: Linear,
    pub h_6_mlp_c_proj: Linear,
    pub h_6_mlp_dropout: Dropout,
    pub h_7_attn_c_attn: Linear,
    pub h_7_attn_c_proj: Linear,
    pub h_7_attn_resid_dropout: Dropout,
    pub h_7_ln_1: LayerNorm,
    pub h_7_ln_2: LayerNorm,
    pub h_7_mlp_act: candle_nn::Activation,
    pub h_7_mlp_c_fc: Linear,
    pub h_7_mlp_c_proj: Linear,
    pub h_7_mlp_dropout: Dropout,
    pub h_8_attn_c_attn: Linear,
    pub h_8_attn_c_proj: Linear,
    pub h_8_attn_resid_dropout: Dropout,
    pub h_8_ln_1: LayerNorm,
    pub h_8_ln_2: LayerNorm,
    pub h_8_mlp_act: candle_nn::Activation,
    pub h_8_mlp_c_fc: Linear,
    pub h_8_mlp_c_proj: Linear,
    pub h_8_mlp_dropout: Dropout,
    pub h_9_attn_c_attn: Linear,
    pub h_9_attn_c_proj: Linear,
    pub h_9_attn_resid_dropout: Dropout,
    pub h_9_ln_1: LayerNorm,
    pub h_9_ln_2: LayerNorm,
    pub h_9_mlp_act: candle_nn::Activation,
    pub h_9_mlp_c_fc: Linear,
    pub h_9_mlp_c_proj: Linear,
    pub h_9_mlp_dropout: Dropout,
    pub ln_f: LayerNorm,
    pub wpe: Embedding,
    pub checker: Option<PyChecker>,
}

impl MyModel {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let drop = Dropout::new();
        let h_0_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.0.attn.c_attn"))?;
        let h_0_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.0.attn.c_proj"))?;
        let h_0_attn_resid_dropout = Dropout::new();
        let h_0_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.0.ln_1"))?;
        let h_0_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.0.ln_2"))?;
        let h_0_mlp_act = candle_nn::Activation::NewGelu;
        let h_0_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.0.mlp.c_fc"))?;
        let h_0_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.0.mlp.c_proj"))?;
        let h_0_mlp_dropout = Dropout::new();
        let h_1_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.1.attn.c_attn"))?;
        let h_1_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.1.attn.c_proj"))?;
        let h_1_attn_resid_dropout = Dropout::new();
        let h_1_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.1.ln_1"))?;
        let h_1_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.1.ln_2"))?;
        let h_1_mlp_act = candle_nn::Activation::NewGelu;
        let h_1_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.1.mlp.c_fc"))?;
        let h_1_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.1.mlp.c_proj"))?;
        let h_1_mlp_dropout = Dropout::new();
        let h_10_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.10.attn.c_attn"))?;
        let h_10_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.10.attn.c_proj"))?;
        let h_10_attn_resid_dropout = Dropout::new();
        let h_10_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.10.ln_1"))?;
        let h_10_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.10.ln_2"))?;
        let h_10_mlp_act = candle_nn::Activation::NewGelu;
        let h_10_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.10.mlp.c_fc"))?;
        let h_10_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.10.mlp.c_proj"))?;
        let h_10_mlp_dropout = Dropout::new();
        let h_11_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.11.attn.c_attn"))?;
        let h_11_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.11.attn.c_proj"))?;
        let h_11_attn_resid_dropout = Dropout::new();
        let h_11_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.11.ln_1"))?;
        let h_11_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.11.ln_2"))?;
        let h_11_mlp_act = candle_nn::Activation::NewGelu;
        let h_11_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.11.mlp.c_fc"))?;
        let h_11_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.11.mlp.c_proj"))?;
        let h_11_mlp_dropout = Dropout::new();
        let h_12_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.12.attn.c_attn"))?;
        let h_12_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.12.attn.c_proj"))?;
        let h_12_attn_resid_dropout = Dropout::new();
        let h_12_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.12.ln_1"))?;
        let h_12_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.12.ln_2"))?;
        let h_12_mlp_act = candle_nn::Activation::NewGelu;
        let h_12_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.12.mlp.c_fc"))?;
        let h_12_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.12.mlp.c_proj"))?;
        let h_12_mlp_dropout = Dropout::new();
        let h_13_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.13.attn.c_attn"))?;
        let h_13_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.13.attn.c_proj"))?;
        let h_13_attn_resid_dropout = Dropout::new();
        let h_13_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.13.ln_1"))?;
        let h_13_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.13.ln_2"))?;
        let h_13_mlp_act = candle_nn::Activation::NewGelu;
        let h_13_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.13.mlp.c_fc"))?;
        let h_13_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.13.mlp.c_proj"))?;
        let h_13_mlp_dropout = Dropout::new();
        let h_14_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.14.attn.c_attn"))?;
        let h_14_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.14.attn.c_proj"))?;
        let h_14_attn_resid_dropout = Dropout::new();
        let h_14_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.14.ln_1"))?;
        let h_14_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.14.ln_2"))?;
        let h_14_mlp_act = candle_nn::Activation::NewGelu;
        let h_14_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.14.mlp.c_fc"))?;
        let h_14_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.14.mlp.c_proj"))?;
        let h_14_mlp_dropout = Dropout::new();
        let h_15_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.15.attn.c_attn"))?;
        let h_15_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.15.attn.c_proj"))?;
        let h_15_attn_resid_dropout = Dropout::new();
        let h_15_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.15.ln_1"))?;
        let h_15_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.15.ln_2"))?;
        let h_15_mlp_act = candle_nn::Activation::NewGelu;
        let h_15_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.15.mlp.c_fc"))?;
        let h_15_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.15.mlp.c_proj"))?;
        let h_15_mlp_dropout = Dropout::new();
        let h_16_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.16.attn.c_attn"))?;
        let h_16_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.16.attn.c_proj"))?;
        let h_16_attn_resid_dropout = Dropout::new();
        let h_16_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.16.ln_1"))?;
        let h_16_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.16.ln_2"))?;
        let h_16_mlp_act = candle_nn::Activation::NewGelu;
        let h_16_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.16.mlp.c_fc"))?;
        let h_16_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.16.mlp.c_proj"))?;
        let h_16_mlp_dropout = Dropout::new();
        let h_17_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.17.attn.c_attn"))?;
        let h_17_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.17.attn.c_proj"))?;
        let h_17_attn_resid_dropout = Dropout::new();
        let h_17_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.17.ln_1"))?;
        let h_17_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.17.ln_2"))?;
        let h_17_mlp_act = candle_nn::Activation::NewGelu;
        let h_17_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.17.mlp.c_fc"))?;
        let h_17_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.17.mlp.c_proj"))?;
        let h_17_mlp_dropout = Dropout::new();
        let h_18_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.18.attn.c_attn"))?;
        let h_18_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.18.attn.c_proj"))?;
        let h_18_attn_resid_dropout = Dropout::new();
        let h_18_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.18.ln_1"))?;
        let h_18_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.18.ln_2"))?;
        let h_18_mlp_act = candle_nn::Activation::NewGelu;
        let h_18_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.18.mlp.c_fc"))?;
        let h_18_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.18.mlp.c_proj"))?;
        let h_18_mlp_dropout = Dropout::new();
        let h_19_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.19.attn.c_attn"))?;
        let h_19_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.19.attn.c_proj"))?;
        let h_19_attn_resid_dropout = Dropout::new();
        let h_19_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.19.ln_1"))?;
        let h_19_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.19.ln_2"))?;
        let h_19_mlp_act = candle_nn::Activation::NewGelu;
        let h_19_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.19.mlp.c_fc"))?;
        let h_19_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.19.mlp.c_proj"))?;
        let h_19_mlp_dropout = Dropout::new();
        let h_2_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.2.attn.c_attn"))?;
        let h_2_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.2.attn.c_proj"))?;
        let h_2_attn_resid_dropout = Dropout::new();
        let h_2_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.2.ln_1"))?;
        let h_2_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.2.ln_2"))?;
        let h_2_mlp_act = candle_nn::Activation::NewGelu;
        let h_2_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.2.mlp.c_fc"))?;
        let h_2_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.2.mlp.c_proj"))?;
        let h_2_mlp_dropout = Dropout::new();
        let h_20_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.20.attn.c_attn"))?;
        let h_20_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.20.attn.c_proj"))?;
        let h_20_attn_resid_dropout = Dropout::new();
        let h_20_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.20.ln_1"))?;
        let h_20_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.20.ln_2"))?;
        let h_20_mlp_act = candle_nn::Activation::NewGelu;
        let h_20_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.20.mlp.c_fc"))?;
        let h_20_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.20.mlp.c_proj"))?;
        let h_20_mlp_dropout = Dropout::new();
        let h_21_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.21.attn.c_attn"))?;
        let h_21_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.21.attn.c_proj"))?;
        let h_21_attn_resid_dropout = Dropout::new();
        let h_21_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.21.ln_1"))?;
        let h_21_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.21.ln_2"))?;
        let h_21_mlp_act = candle_nn::Activation::NewGelu;
        let h_21_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.21.mlp.c_fc"))?;
        let h_21_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.21.mlp.c_proj"))?;
        let h_21_mlp_dropout = Dropout::new();
        let h_22_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.22.attn.c_attn"))?;
        let h_22_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.22.attn.c_proj"))?;
        let h_22_attn_resid_dropout = Dropout::new();
        let h_22_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.22.ln_1"))?;
        let h_22_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.22.ln_2"))?;
        let h_22_mlp_act = candle_nn::Activation::NewGelu;
        let h_22_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.22.mlp.c_fc"))?;
        let h_22_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.22.mlp.c_proj"))?;
        let h_22_mlp_dropout = Dropout::new();
        let h_23_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.23.attn.c_attn"))?;
        let h_23_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.23.attn.c_proj"))?;
        let h_23_attn_resid_dropout = Dropout::new();
        let h_23_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.23.ln_1"))?;
        let h_23_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.23.ln_2"))?;
        let h_23_mlp_act = candle_nn::Activation::NewGelu;
        let h_23_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.23.mlp.c_fc"))?;
        let h_23_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.23.mlp.c_proj"))?;
        let h_23_mlp_dropout = Dropout::new();
        let h_3_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.3.attn.c_attn"))?;
        let h_3_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.3.attn.c_proj"))?;
        let h_3_attn_resid_dropout = Dropout::new();
        let h_3_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.3.ln_1"))?;
        let h_3_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.3.ln_2"))?;
        let h_3_mlp_act = candle_nn::Activation::NewGelu;
        let h_3_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.3.mlp.c_fc"))?;
        let h_3_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.3.mlp.c_proj"))?;
        let h_3_mlp_dropout = Dropout::new();
        let h_4_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.4.attn.c_attn"))?;
        let h_4_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.4.attn.c_proj"))?;
        let h_4_attn_resid_dropout = Dropout::new();
        let h_4_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.4.ln_1"))?;
        let h_4_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.4.ln_2"))?;
        let h_4_mlp_act = candle_nn::Activation::NewGelu;
        let h_4_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.4.mlp.c_fc"))?;
        let h_4_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.4.mlp.c_proj"))?;
        let h_4_mlp_dropout = Dropout::new();
        let h_5_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.5.attn.c_attn"))?;
        let h_5_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.5.attn.c_proj"))?;
        let h_5_attn_resid_dropout = Dropout::new();
        let h_5_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.5.ln_1"))?;
        let h_5_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.5.ln_2"))?;
        let h_5_mlp_act = candle_nn::Activation::NewGelu;
        let h_5_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.5.mlp.c_fc"))?;
        let h_5_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.5.mlp.c_proj"))?;
        let h_5_mlp_dropout = Dropout::new();
        let h_6_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.6.attn.c_attn"))?;
        let h_6_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.6.attn.c_proj"))?;
        let h_6_attn_resid_dropout = Dropout::new();
        let h_6_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.6.ln_1"))?;
        let h_6_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.6.ln_2"))?;
        let h_6_mlp_act = candle_nn::Activation::NewGelu;
        let h_6_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.6.mlp.c_fc"))?;
        let h_6_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.6.mlp.c_proj"))?;
        let h_6_mlp_dropout = Dropout::new();
        let h_7_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.7.attn.c_attn"))?;
        let h_7_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.7.attn.c_proj"))?;
        let h_7_attn_resid_dropout = Dropout::new();
        let h_7_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.7.ln_1"))?;
        let h_7_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.7.ln_2"))?;
        let h_7_mlp_act = candle_nn::Activation::NewGelu;
        let h_7_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.7.mlp.c_fc"))?;
        let h_7_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.7.mlp.c_proj"))?;
        let h_7_mlp_dropout = Dropout::new();
        let h_8_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.8.attn.c_attn"))?;
        let h_8_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.8.attn.c_proj"))?;
        let h_8_attn_resid_dropout = Dropout::new();
        let h_8_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.8.ln_1"))?;
        let h_8_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.8.ln_2"))?;
        let h_8_mlp_act = candle_nn::Activation::NewGelu;
        let h_8_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.8.mlp.c_fc"))?;
        let h_8_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.8.mlp.c_proj"))?;
        let h_8_mlp_dropout = Dropout::new();
        let h_9_attn_c_attn = candle_nn::linear(0, 0, vb.pp("h.9.attn.c_attn"))?;
        let h_9_attn_c_proj = candle_nn::linear(0, 0, vb.pp("h.9.attn.c_proj"))?;
        let h_9_attn_resid_dropout = Dropout::new();
        let h_9_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.9.ln_1"))?;
        let h_9_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.9.ln_2"))?;
        let h_9_mlp_act = candle_nn::Activation::NewGelu;
        let h_9_mlp_c_fc = candle_nn::linear(0, 0, vb.pp("h.9.mlp.c_fc"))?;
        let h_9_mlp_c_proj = candle_nn::linear(0, 0, vb.pp("h.9.mlp.c_proj"))?;
        let h_9_mlp_dropout = Dropout::new();
        let ln_f = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("ln_f"))?;
        let wpe = candle_nn::embedding(8196, 1024, vb.pp("wpe"))?;

        Ok(Self {
            drop,
            h_0_attn_c_attn,
            h_0_attn_c_proj,
            h_0_attn_resid_dropout,
            h_0_ln_1,
            h_0_ln_2,
            h_0_mlp_act,
            h_0_mlp_c_fc,
            h_0_mlp_c_proj,
            h_0_mlp_dropout,
            h_1_attn_c_attn,
            h_1_attn_c_proj,
            h_1_attn_resid_dropout,
            h_1_ln_1,
            h_1_ln_2,
            h_1_mlp_act,
            h_1_mlp_c_fc,
            h_1_mlp_c_proj,
            h_1_mlp_dropout,
            h_10_attn_c_attn,
            h_10_attn_c_proj,
            h_10_attn_resid_dropout,
            h_10_ln_1,
            h_10_ln_2,
            h_10_mlp_act,
            h_10_mlp_c_fc,
            h_10_mlp_c_proj,
            h_10_mlp_dropout,
            h_11_attn_c_attn,
            h_11_attn_c_proj,
            h_11_attn_resid_dropout,
            h_11_ln_1,
            h_11_ln_2,
            h_11_mlp_act,
            h_11_mlp_c_fc,
            h_11_mlp_c_proj,
            h_11_mlp_dropout,
            h_12_attn_c_attn,
            h_12_attn_c_proj,
            h_12_attn_resid_dropout,
            h_12_ln_1,
            h_12_ln_2,
            h_12_mlp_act,
            h_12_mlp_c_fc,
            h_12_mlp_c_proj,
            h_12_mlp_dropout,
            h_13_attn_c_attn,
            h_13_attn_c_proj,
            h_13_attn_resid_dropout,
            h_13_ln_1,
            h_13_ln_2,
            h_13_mlp_act,
            h_13_mlp_c_fc,
            h_13_mlp_c_proj,
            h_13_mlp_dropout,
            h_14_attn_c_attn,
            h_14_attn_c_proj,
            h_14_attn_resid_dropout,
            h_14_ln_1,
            h_14_ln_2,
            h_14_mlp_act,
            h_14_mlp_c_fc,
            h_14_mlp_c_proj,
            h_14_mlp_dropout,
            h_15_attn_c_attn,
            h_15_attn_c_proj,
            h_15_attn_resid_dropout,
            h_15_ln_1,
            h_15_ln_2,
            h_15_mlp_act,
            h_15_mlp_c_fc,
            h_15_mlp_c_proj,
            h_15_mlp_dropout,
            h_16_attn_c_attn,
            h_16_attn_c_proj,
            h_16_attn_resid_dropout,
            h_16_ln_1,
            h_16_ln_2,
            h_16_mlp_act,
            h_16_mlp_c_fc,
            h_16_mlp_c_proj,
            h_16_mlp_dropout,
            h_17_attn_c_attn,
            h_17_attn_c_proj,
            h_17_attn_resid_dropout,
            h_17_ln_1,
            h_17_ln_2,
            h_17_mlp_act,
            h_17_mlp_c_fc,
            h_17_mlp_c_proj,
            h_17_mlp_dropout,
            h_18_attn_c_attn,
            h_18_attn_c_proj,
            h_18_attn_resid_dropout,
            h_18_ln_1,
            h_18_ln_2,
            h_18_mlp_act,
            h_18_mlp_c_fc,
            h_18_mlp_c_proj,
            h_18_mlp_dropout,
            h_19_attn_c_attn,
            h_19_attn_c_proj,
            h_19_attn_resid_dropout,
            h_19_ln_1,
            h_19_ln_2,
            h_19_mlp_act,
            h_19_mlp_c_fc,
            h_19_mlp_c_proj,
            h_19_mlp_dropout,
            h_2_attn_c_attn,
            h_2_attn_c_proj,
            h_2_attn_resid_dropout,
            h_2_ln_1,
            h_2_ln_2,
            h_2_mlp_act,
            h_2_mlp_c_fc,
            h_2_mlp_c_proj,
            h_2_mlp_dropout,
            h_20_attn_c_attn,
            h_20_attn_c_proj,
            h_20_attn_resid_dropout,
            h_20_ln_1,
            h_20_ln_2,
            h_20_mlp_act,
            h_20_mlp_c_fc,
            h_20_mlp_c_proj,
            h_20_mlp_dropout,
            h_21_attn_c_attn,
            h_21_attn_c_proj,
            h_21_attn_resid_dropout,
            h_21_ln_1,
            h_21_ln_2,
            h_21_mlp_act,
            h_21_mlp_c_fc,
            h_21_mlp_c_proj,
            h_21_mlp_dropout,
            h_22_attn_c_attn,
            h_22_attn_c_proj,
            h_22_attn_resid_dropout,
            h_22_ln_1,
            h_22_ln_2,
            h_22_mlp_act,
            h_22_mlp_c_fc,
            h_22_mlp_c_proj,
            h_22_mlp_dropout,
            h_23_attn_c_attn,
            h_23_attn_c_proj,
            h_23_attn_resid_dropout,
            h_23_ln_1,
            h_23_ln_2,
            h_23_mlp_act,
            h_23_mlp_c_fc,
            h_23_mlp_c_proj,
            h_23_mlp_dropout,
            h_3_attn_c_attn,
            h_3_attn_c_proj,
            h_3_attn_resid_dropout,
            h_3_ln_1,
            h_3_ln_2,
            h_3_mlp_act,
            h_3_mlp_c_fc,
            h_3_mlp_c_proj,
            h_3_mlp_dropout,
            h_4_attn_c_attn,
            h_4_attn_c_proj,
            h_4_attn_resid_dropout,
            h_4_ln_1,
            h_4_ln_2,
            h_4_mlp_act,
            h_4_mlp_c_fc,
            h_4_mlp_c_proj,
            h_4_mlp_dropout,
            h_5_attn_c_attn,
            h_5_attn_c_proj,
            h_5_attn_resid_dropout,
            h_5_ln_1,
            h_5_ln_2,
            h_5_mlp_act,
            h_5_mlp_c_fc,
            h_5_mlp_c_proj,
            h_5_mlp_dropout,
            h_6_attn_c_attn,
            h_6_attn_c_proj,
            h_6_attn_resid_dropout,
            h_6_ln_1,
            h_6_ln_2,
            h_6_mlp_act,
            h_6_mlp_c_fc,
            h_6_mlp_c_proj,
            h_6_mlp_dropout,
            h_7_attn_c_attn,
            h_7_attn_c_proj,
            h_7_attn_resid_dropout,
            h_7_ln_1,
            h_7_ln_2,
            h_7_mlp_act,
            h_7_mlp_c_fc,
            h_7_mlp_c_proj,
            h_7_mlp_dropout,
            h_8_attn_c_attn,
            h_8_attn_c_proj,
            h_8_attn_resid_dropout,
            h_8_ln_1,
            h_8_ln_2,
            h_8_mlp_act,
            h_8_mlp_c_fc,
            h_8_mlp_c_proj,
            h_8_mlp_dropout,
            h_9_attn_c_attn,
            h_9_attn_c_proj,
            h_9_attn_resid_dropout,
            h_9_ln_1,
            h_9_ln_2,
            h_9_mlp_act,
            h_9_mlp_c_fc,
            h_9_mlp_c_proj,
            h_9_mlp_dropout,
            ln_f,
            wpe,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: drop
        x = self.drop.forward(&x)?;
        py_check!(self.checker, "drop", &x);

        // Layer: h.0.attn.c_attn
        x = self.h_0_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.0.attn.c_attn", &x);

        // Layer: h.0.attn.c_proj
        x = self.h_0_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.0.attn.c_proj", &x);

        // Layer: h.0.attn.resid_dropout
        x = self.h_0_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.0.attn.resid_dropout", &x);

        // Layer: h.0.ln_1
        x = self.h_0_ln_1.forward(&x)?;
        py_check!(self.checker, "h.0.ln_1", &x);

        // Layer: h.0.ln_2
        x = self.h_0_ln_2.forward(&x)?;
        py_check!(self.checker, "h.0.ln_2", &x);

        // Layer: h.0.mlp.act
        x = self.h_0_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.0.mlp.act", &x);

        // Layer: h.0.mlp.c_fc
        x = self.h_0_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.0.mlp.c_fc", &x);

        // Layer: h.0.mlp.c_proj
        x = self.h_0_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.0.mlp.c_proj", &x);

        // Layer: h.0.mlp.dropout
        x = self.h_0_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.0.mlp.dropout", &x);

        // Layer: h.1.attn.c_attn
        x = self.h_1_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.1.attn.c_attn", &x);

        // Layer: h.1.attn.c_proj
        x = self.h_1_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.1.attn.c_proj", &x);

        // Layer: h.1.attn.resid_dropout
        x = self.h_1_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.1.attn.resid_dropout", &x);

        // Layer: h.1.ln_1
        x = self.h_1_ln_1.forward(&x)?;
        py_check!(self.checker, "h.1.ln_1", &x);

        // Layer: h.1.ln_2
        x = self.h_1_ln_2.forward(&x)?;
        py_check!(self.checker, "h.1.ln_2", &x);

        // Layer: h.1.mlp.act
        x = self.h_1_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.1.mlp.act", &x);

        // Layer: h.1.mlp.c_fc
        x = self.h_1_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.1.mlp.c_fc", &x);

        // Layer: h.1.mlp.c_proj
        x = self.h_1_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.1.mlp.c_proj", &x);

        // Layer: h.1.mlp.dropout
        x = self.h_1_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.1.mlp.dropout", &x);

        // Layer: h.10.attn.c_attn
        x = self.h_10_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.10.attn.c_attn", &x);

        // Layer: h.10.attn.c_proj
        x = self.h_10_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.10.attn.c_proj", &x);

        // Layer: h.10.attn.resid_dropout
        x = self.h_10_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.10.attn.resid_dropout", &x);

        // Layer: h.10.ln_1
        x = self.h_10_ln_1.forward(&x)?;
        py_check!(self.checker, "h.10.ln_1", &x);

        // Layer: h.10.ln_2
        x = self.h_10_ln_2.forward(&x)?;
        py_check!(self.checker, "h.10.ln_2", &x);

        // Layer: h.10.mlp.act
        x = self.h_10_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.10.mlp.act", &x);

        // Layer: h.10.mlp.c_fc
        x = self.h_10_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.10.mlp.c_fc", &x);

        // Layer: h.10.mlp.c_proj
        x = self.h_10_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.10.mlp.c_proj", &x);

        // Layer: h.10.mlp.dropout
        x = self.h_10_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.10.mlp.dropout", &x);

        // Layer: h.11.attn.c_attn
        x = self.h_11_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.11.attn.c_attn", &x);

        // Layer: h.11.attn.c_proj
        x = self.h_11_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.11.attn.c_proj", &x);

        // Layer: h.11.attn.resid_dropout
        x = self.h_11_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.11.attn.resid_dropout", &x);

        // Layer: h.11.ln_1
        x = self.h_11_ln_1.forward(&x)?;
        py_check!(self.checker, "h.11.ln_1", &x);

        // Layer: h.11.ln_2
        x = self.h_11_ln_2.forward(&x)?;
        py_check!(self.checker, "h.11.ln_2", &x);

        // Layer: h.11.mlp.act
        x = self.h_11_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.11.mlp.act", &x);

        // Layer: h.11.mlp.c_fc
        x = self.h_11_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.11.mlp.c_fc", &x);

        // Layer: h.11.mlp.c_proj
        x = self.h_11_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.11.mlp.c_proj", &x);

        // Layer: h.11.mlp.dropout
        x = self.h_11_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.11.mlp.dropout", &x);

        // Layer: h.12.attn.c_attn
        x = self.h_12_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.12.attn.c_attn", &x);

        // Layer: h.12.attn.c_proj
        x = self.h_12_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.12.attn.c_proj", &x);

        // Layer: h.12.attn.resid_dropout
        x = self.h_12_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.12.attn.resid_dropout", &x);

        // Layer: h.12.ln_1
        x = self.h_12_ln_1.forward(&x)?;
        py_check!(self.checker, "h.12.ln_1", &x);

        // Layer: h.12.ln_2
        x = self.h_12_ln_2.forward(&x)?;
        py_check!(self.checker, "h.12.ln_2", &x);

        // Layer: h.12.mlp.act
        x = self.h_12_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.12.mlp.act", &x);

        // Layer: h.12.mlp.c_fc
        x = self.h_12_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.12.mlp.c_fc", &x);

        // Layer: h.12.mlp.c_proj
        x = self.h_12_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.12.mlp.c_proj", &x);

        // Layer: h.12.mlp.dropout
        x = self.h_12_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.12.mlp.dropout", &x);

        // Layer: h.13.attn.c_attn
        x = self.h_13_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.13.attn.c_attn", &x);

        // Layer: h.13.attn.c_proj
        x = self.h_13_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.13.attn.c_proj", &x);

        // Layer: h.13.attn.resid_dropout
        x = self.h_13_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.13.attn.resid_dropout", &x);

        // Layer: h.13.ln_1
        x = self.h_13_ln_1.forward(&x)?;
        py_check!(self.checker, "h.13.ln_1", &x);

        // Layer: h.13.ln_2
        x = self.h_13_ln_2.forward(&x)?;
        py_check!(self.checker, "h.13.ln_2", &x);

        // Layer: h.13.mlp.act
        x = self.h_13_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.13.mlp.act", &x);

        // Layer: h.13.mlp.c_fc
        x = self.h_13_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.13.mlp.c_fc", &x);

        // Layer: h.13.mlp.c_proj
        x = self.h_13_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.13.mlp.c_proj", &x);

        // Layer: h.13.mlp.dropout
        x = self.h_13_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.13.mlp.dropout", &x);

        // Layer: h.14.attn.c_attn
        x = self.h_14_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.14.attn.c_attn", &x);

        // Layer: h.14.attn.c_proj
        x = self.h_14_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.14.attn.c_proj", &x);

        // Layer: h.14.attn.resid_dropout
        x = self.h_14_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.14.attn.resid_dropout", &x);

        // Layer: h.14.ln_1
        x = self.h_14_ln_1.forward(&x)?;
        py_check!(self.checker, "h.14.ln_1", &x);

        // Layer: h.14.ln_2
        x = self.h_14_ln_2.forward(&x)?;
        py_check!(self.checker, "h.14.ln_2", &x);

        // Layer: h.14.mlp.act
        x = self.h_14_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.14.mlp.act", &x);

        // Layer: h.14.mlp.c_fc
        x = self.h_14_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.14.mlp.c_fc", &x);

        // Layer: h.14.mlp.c_proj
        x = self.h_14_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.14.mlp.c_proj", &x);

        // Layer: h.14.mlp.dropout
        x = self.h_14_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.14.mlp.dropout", &x);

        // Layer: h.15.attn.c_attn
        x = self.h_15_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.15.attn.c_attn", &x);

        // Layer: h.15.attn.c_proj
        x = self.h_15_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.15.attn.c_proj", &x);

        // Layer: h.15.attn.resid_dropout
        x = self.h_15_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.15.attn.resid_dropout", &x);

        // Layer: h.15.ln_1
        x = self.h_15_ln_1.forward(&x)?;
        py_check!(self.checker, "h.15.ln_1", &x);

        // Layer: h.15.ln_2
        x = self.h_15_ln_2.forward(&x)?;
        py_check!(self.checker, "h.15.ln_2", &x);

        // Layer: h.15.mlp.act
        x = self.h_15_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.15.mlp.act", &x);

        // Layer: h.15.mlp.c_fc
        x = self.h_15_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.15.mlp.c_fc", &x);

        // Layer: h.15.mlp.c_proj
        x = self.h_15_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.15.mlp.c_proj", &x);

        // Layer: h.15.mlp.dropout
        x = self.h_15_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.15.mlp.dropout", &x);

        // Layer: h.16.attn.c_attn
        x = self.h_16_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.16.attn.c_attn", &x);

        // Layer: h.16.attn.c_proj
        x = self.h_16_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.16.attn.c_proj", &x);

        // Layer: h.16.attn.resid_dropout
        x = self.h_16_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.16.attn.resid_dropout", &x);

        // Layer: h.16.ln_1
        x = self.h_16_ln_1.forward(&x)?;
        py_check!(self.checker, "h.16.ln_1", &x);

        // Layer: h.16.ln_2
        x = self.h_16_ln_2.forward(&x)?;
        py_check!(self.checker, "h.16.ln_2", &x);

        // Layer: h.16.mlp.act
        x = self.h_16_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.16.mlp.act", &x);

        // Layer: h.16.mlp.c_fc
        x = self.h_16_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.16.mlp.c_fc", &x);

        // Layer: h.16.mlp.c_proj
        x = self.h_16_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.16.mlp.c_proj", &x);

        // Layer: h.16.mlp.dropout
        x = self.h_16_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.16.mlp.dropout", &x);

        // Layer: h.17.attn.c_attn
        x = self.h_17_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.17.attn.c_attn", &x);

        // Layer: h.17.attn.c_proj
        x = self.h_17_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.17.attn.c_proj", &x);

        // Layer: h.17.attn.resid_dropout
        x = self.h_17_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.17.attn.resid_dropout", &x);

        // Layer: h.17.ln_1
        x = self.h_17_ln_1.forward(&x)?;
        py_check!(self.checker, "h.17.ln_1", &x);

        // Layer: h.17.ln_2
        x = self.h_17_ln_2.forward(&x)?;
        py_check!(self.checker, "h.17.ln_2", &x);

        // Layer: h.17.mlp.act
        x = self.h_17_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.17.mlp.act", &x);

        // Layer: h.17.mlp.c_fc
        x = self.h_17_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.17.mlp.c_fc", &x);

        // Layer: h.17.mlp.c_proj
        x = self.h_17_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.17.mlp.c_proj", &x);

        // Layer: h.17.mlp.dropout
        x = self.h_17_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.17.mlp.dropout", &x);

        // Layer: h.18.attn.c_attn
        x = self.h_18_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.18.attn.c_attn", &x);

        // Layer: h.18.attn.c_proj
        x = self.h_18_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.18.attn.c_proj", &x);

        // Layer: h.18.attn.resid_dropout
        x = self.h_18_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.18.attn.resid_dropout", &x);

        // Layer: h.18.ln_1
        x = self.h_18_ln_1.forward(&x)?;
        py_check!(self.checker, "h.18.ln_1", &x);

        // Layer: h.18.ln_2
        x = self.h_18_ln_2.forward(&x)?;
        py_check!(self.checker, "h.18.ln_2", &x);

        // Layer: h.18.mlp.act
        x = self.h_18_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.18.mlp.act", &x);

        // Layer: h.18.mlp.c_fc
        x = self.h_18_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.18.mlp.c_fc", &x);

        // Layer: h.18.mlp.c_proj
        x = self.h_18_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.18.mlp.c_proj", &x);

        // Layer: h.18.mlp.dropout
        x = self.h_18_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.18.mlp.dropout", &x);

        // Layer: h.19.attn.c_attn
        x = self.h_19_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.19.attn.c_attn", &x);

        // Layer: h.19.attn.c_proj
        x = self.h_19_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.19.attn.c_proj", &x);

        // Layer: h.19.attn.resid_dropout
        x = self.h_19_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.19.attn.resid_dropout", &x);

        // Layer: h.19.ln_1
        x = self.h_19_ln_1.forward(&x)?;
        py_check!(self.checker, "h.19.ln_1", &x);

        // Layer: h.19.ln_2
        x = self.h_19_ln_2.forward(&x)?;
        py_check!(self.checker, "h.19.ln_2", &x);

        // Layer: h.19.mlp.act
        x = self.h_19_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.19.mlp.act", &x);

        // Layer: h.19.mlp.c_fc
        x = self.h_19_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.19.mlp.c_fc", &x);

        // Layer: h.19.mlp.c_proj
        x = self.h_19_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.19.mlp.c_proj", &x);

        // Layer: h.19.mlp.dropout
        x = self.h_19_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.19.mlp.dropout", &x);

        // Layer: h.2.attn.c_attn
        x = self.h_2_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.2.attn.c_attn", &x);

        // Layer: h.2.attn.c_proj
        x = self.h_2_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.2.attn.c_proj", &x);

        // Layer: h.2.attn.resid_dropout
        x = self.h_2_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.2.attn.resid_dropout", &x);

        // Layer: h.2.ln_1
        x = self.h_2_ln_1.forward(&x)?;
        py_check!(self.checker, "h.2.ln_1", &x);

        // Layer: h.2.ln_2
        x = self.h_2_ln_2.forward(&x)?;
        py_check!(self.checker, "h.2.ln_2", &x);

        // Layer: h.2.mlp.act
        x = self.h_2_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.2.mlp.act", &x);

        // Layer: h.2.mlp.c_fc
        x = self.h_2_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.2.mlp.c_fc", &x);

        // Layer: h.2.mlp.c_proj
        x = self.h_2_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.2.mlp.c_proj", &x);

        // Layer: h.2.mlp.dropout
        x = self.h_2_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.2.mlp.dropout", &x);

        // Layer: h.20.attn.c_attn
        x = self.h_20_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.20.attn.c_attn", &x);

        // Layer: h.20.attn.c_proj
        x = self.h_20_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.20.attn.c_proj", &x);

        // Layer: h.20.attn.resid_dropout
        x = self.h_20_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.20.attn.resid_dropout", &x);

        // Layer: h.20.ln_1
        x = self.h_20_ln_1.forward(&x)?;
        py_check!(self.checker, "h.20.ln_1", &x);

        // Layer: h.20.ln_2
        x = self.h_20_ln_2.forward(&x)?;
        py_check!(self.checker, "h.20.ln_2", &x);

        // Layer: h.20.mlp.act
        x = self.h_20_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.20.mlp.act", &x);

        // Layer: h.20.mlp.c_fc
        x = self.h_20_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.20.mlp.c_fc", &x);

        // Layer: h.20.mlp.c_proj
        x = self.h_20_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.20.mlp.c_proj", &x);

        // Layer: h.20.mlp.dropout
        x = self.h_20_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.20.mlp.dropout", &x);

        // Layer: h.21.attn.c_attn
        x = self.h_21_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.21.attn.c_attn", &x);

        // Layer: h.21.attn.c_proj
        x = self.h_21_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.21.attn.c_proj", &x);

        // Layer: h.21.attn.resid_dropout
        x = self.h_21_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.21.attn.resid_dropout", &x);

        // Layer: h.21.ln_1
        x = self.h_21_ln_1.forward(&x)?;
        py_check!(self.checker, "h.21.ln_1", &x);

        // Layer: h.21.ln_2
        x = self.h_21_ln_2.forward(&x)?;
        py_check!(self.checker, "h.21.ln_2", &x);

        // Layer: h.21.mlp.act
        x = self.h_21_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.21.mlp.act", &x);

        // Layer: h.21.mlp.c_fc
        x = self.h_21_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.21.mlp.c_fc", &x);

        // Layer: h.21.mlp.c_proj
        x = self.h_21_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.21.mlp.c_proj", &x);

        // Layer: h.21.mlp.dropout
        x = self.h_21_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.21.mlp.dropout", &x);

        // Layer: h.22.attn.c_attn
        x = self.h_22_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.22.attn.c_attn", &x);

        // Layer: h.22.attn.c_proj
        x = self.h_22_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.22.attn.c_proj", &x);

        // Layer: h.22.attn.resid_dropout
        x = self.h_22_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.22.attn.resid_dropout", &x);

        // Layer: h.22.ln_1
        x = self.h_22_ln_1.forward(&x)?;
        py_check!(self.checker, "h.22.ln_1", &x);

        // Layer: h.22.ln_2
        x = self.h_22_ln_2.forward(&x)?;
        py_check!(self.checker, "h.22.ln_2", &x);

        // Layer: h.22.mlp.act
        x = self.h_22_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.22.mlp.act", &x);

        // Layer: h.22.mlp.c_fc
        x = self.h_22_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.22.mlp.c_fc", &x);

        // Layer: h.22.mlp.c_proj
        x = self.h_22_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.22.mlp.c_proj", &x);

        // Layer: h.22.mlp.dropout
        x = self.h_22_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.22.mlp.dropout", &x);

        // Layer: h.23.attn.c_attn
        x = self.h_23_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.23.attn.c_attn", &x);

        // Layer: h.23.attn.c_proj
        x = self.h_23_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.23.attn.c_proj", &x);

        // Layer: h.23.attn.resid_dropout
        x = self.h_23_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.23.attn.resid_dropout", &x);

        // Layer: h.23.ln_1
        x = self.h_23_ln_1.forward(&x)?;
        py_check!(self.checker, "h.23.ln_1", &x);

        // Layer: h.23.ln_2
        x = self.h_23_ln_2.forward(&x)?;
        py_check!(self.checker, "h.23.ln_2", &x);

        // Layer: h.23.mlp.act
        x = self.h_23_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.23.mlp.act", &x);

        // Layer: h.23.mlp.c_fc
        x = self.h_23_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.23.mlp.c_fc", &x);

        // Layer: h.23.mlp.c_proj
        x = self.h_23_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.23.mlp.c_proj", &x);

        // Layer: h.23.mlp.dropout
        x = self.h_23_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.23.mlp.dropout", &x);

        // Layer: h.3.attn.c_attn
        x = self.h_3_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.3.attn.c_attn", &x);

        // Layer: h.3.attn.c_proj
        x = self.h_3_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.3.attn.c_proj", &x);

        // Layer: h.3.attn.resid_dropout
        x = self.h_3_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.3.attn.resid_dropout", &x);

        // Layer: h.3.ln_1
        x = self.h_3_ln_1.forward(&x)?;
        py_check!(self.checker, "h.3.ln_1", &x);

        // Layer: h.3.ln_2
        x = self.h_3_ln_2.forward(&x)?;
        py_check!(self.checker, "h.3.ln_2", &x);

        // Layer: h.3.mlp.act
        x = self.h_3_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.3.mlp.act", &x);

        // Layer: h.3.mlp.c_fc
        x = self.h_3_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.3.mlp.c_fc", &x);

        // Layer: h.3.mlp.c_proj
        x = self.h_3_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.3.mlp.c_proj", &x);

        // Layer: h.3.mlp.dropout
        x = self.h_3_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.3.mlp.dropout", &x);

        // Layer: h.4.attn.c_attn
        x = self.h_4_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.4.attn.c_attn", &x);

        // Layer: h.4.attn.c_proj
        x = self.h_4_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.4.attn.c_proj", &x);

        // Layer: h.4.attn.resid_dropout
        x = self.h_4_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.4.attn.resid_dropout", &x);

        // Layer: h.4.ln_1
        x = self.h_4_ln_1.forward(&x)?;
        py_check!(self.checker, "h.4.ln_1", &x);

        // Layer: h.4.ln_2
        x = self.h_4_ln_2.forward(&x)?;
        py_check!(self.checker, "h.4.ln_2", &x);

        // Layer: h.4.mlp.act
        x = self.h_4_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.4.mlp.act", &x);

        // Layer: h.4.mlp.c_fc
        x = self.h_4_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.4.mlp.c_fc", &x);

        // Layer: h.4.mlp.c_proj
        x = self.h_4_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.4.mlp.c_proj", &x);

        // Layer: h.4.mlp.dropout
        x = self.h_4_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.4.mlp.dropout", &x);

        // Layer: h.5.attn.c_attn
        x = self.h_5_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.5.attn.c_attn", &x);

        // Layer: h.5.attn.c_proj
        x = self.h_5_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.5.attn.c_proj", &x);

        // Layer: h.5.attn.resid_dropout
        x = self.h_5_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.5.attn.resid_dropout", &x);

        // Layer: h.5.ln_1
        x = self.h_5_ln_1.forward(&x)?;
        py_check!(self.checker, "h.5.ln_1", &x);

        // Layer: h.5.ln_2
        x = self.h_5_ln_2.forward(&x)?;
        py_check!(self.checker, "h.5.ln_2", &x);

        // Layer: h.5.mlp.act
        x = self.h_5_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.5.mlp.act", &x);

        // Layer: h.5.mlp.c_fc
        x = self.h_5_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.5.mlp.c_fc", &x);

        // Layer: h.5.mlp.c_proj
        x = self.h_5_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.5.mlp.c_proj", &x);

        // Layer: h.5.mlp.dropout
        x = self.h_5_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.5.mlp.dropout", &x);

        // Layer: h.6.attn.c_attn
        x = self.h_6_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.6.attn.c_attn", &x);

        // Layer: h.6.attn.c_proj
        x = self.h_6_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.6.attn.c_proj", &x);

        // Layer: h.6.attn.resid_dropout
        x = self.h_6_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.6.attn.resid_dropout", &x);

        // Layer: h.6.ln_1
        x = self.h_6_ln_1.forward(&x)?;
        py_check!(self.checker, "h.6.ln_1", &x);

        // Layer: h.6.ln_2
        x = self.h_6_ln_2.forward(&x)?;
        py_check!(self.checker, "h.6.ln_2", &x);

        // Layer: h.6.mlp.act
        x = self.h_6_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.6.mlp.act", &x);

        // Layer: h.6.mlp.c_fc
        x = self.h_6_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.6.mlp.c_fc", &x);

        // Layer: h.6.mlp.c_proj
        x = self.h_6_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.6.mlp.c_proj", &x);

        // Layer: h.6.mlp.dropout
        x = self.h_6_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.6.mlp.dropout", &x);

        // Layer: h.7.attn.c_attn
        x = self.h_7_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.7.attn.c_attn", &x);

        // Layer: h.7.attn.c_proj
        x = self.h_7_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.7.attn.c_proj", &x);

        // Layer: h.7.attn.resid_dropout
        x = self.h_7_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.7.attn.resid_dropout", &x);

        // Layer: h.7.ln_1
        x = self.h_7_ln_1.forward(&x)?;
        py_check!(self.checker, "h.7.ln_1", &x);

        // Layer: h.7.ln_2
        x = self.h_7_ln_2.forward(&x)?;
        py_check!(self.checker, "h.7.ln_2", &x);

        // Layer: h.7.mlp.act
        x = self.h_7_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.7.mlp.act", &x);

        // Layer: h.7.mlp.c_fc
        x = self.h_7_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.7.mlp.c_fc", &x);

        // Layer: h.7.mlp.c_proj
        x = self.h_7_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.7.mlp.c_proj", &x);

        // Layer: h.7.mlp.dropout
        x = self.h_7_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.7.mlp.dropout", &x);

        // Layer: h.8.attn.c_attn
        x = self.h_8_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.8.attn.c_attn", &x);

        // Layer: h.8.attn.c_proj
        x = self.h_8_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.8.attn.c_proj", &x);

        // Layer: h.8.attn.resid_dropout
        x = self.h_8_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.8.attn.resid_dropout", &x);

        // Layer: h.8.ln_1
        x = self.h_8_ln_1.forward(&x)?;
        py_check!(self.checker, "h.8.ln_1", &x);

        // Layer: h.8.ln_2
        x = self.h_8_ln_2.forward(&x)?;
        py_check!(self.checker, "h.8.ln_2", &x);

        // Layer: h.8.mlp.act
        x = self.h_8_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.8.mlp.act", &x);

        // Layer: h.8.mlp.c_fc
        x = self.h_8_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.8.mlp.c_fc", &x);

        // Layer: h.8.mlp.c_proj
        x = self.h_8_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.8.mlp.c_proj", &x);

        // Layer: h.8.mlp.dropout
        x = self.h_8_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.8.mlp.dropout", &x);

        // Layer: h.9.attn.c_attn
        x = self.h_9_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "h.9.attn.c_attn", &x);

        // Layer: h.9.attn.c_proj
        x = self.h_9_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "h.9.attn.c_proj", &x);

        // Layer: h.9.attn.resid_dropout
        x = self.h_9_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "h.9.attn.resid_dropout", &x);

        // Layer: h.9.ln_1
        x = self.h_9_ln_1.forward(&x)?;
        py_check!(self.checker, "h.9.ln_1", &x);

        // Layer: h.9.ln_2
        x = self.h_9_ln_2.forward(&x)?;
        py_check!(self.checker, "h.9.ln_2", &x);

        // Layer: h.9.mlp.act
        x = self.h_9_mlp_act.forward(&x)?;
        py_check!(self.checker, "h.9.mlp.act", &x);

        // Layer: h.9.mlp.c_fc
        x = self.h_9_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "h.9.mlp.c_fc", &x);

        // Layer: h.9.mlp.c_proj
        x = self.h_9_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "h.9.mlp.c_proj", &x);

        // Layer: h.9.mlp.dropout
        x = self.h_9_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "h.9.mlp.dropout", &x);

        // Layer: ln_f
        x = self.ln_f.forward(&x)?;
        py_check!(self.checker, "ln_f", &x);

        // Layer: wpe
        x = self.wpe.forward(&x)?;
        py_check!(self.checker, "wpe", &x);

        Ok(x)
    }
}
