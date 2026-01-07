use candle_core::{Tensor, Result, Device};
use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};
use crate::{PyChecker, py_check};
use crate::gpt2;

pub struct GPT2Model {
    pub lm_head: Linear,
    pub transformer_drop: () /* TODO: Dropout */,
    pub transformer_h_0_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_0_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_0_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_0_ln_1: LayerNorm,
    pub transformer_h_0_ln_2: LayerNorm,
    pub transformer_h_0_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_0_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_0_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_0_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_1_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_1_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_1_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_1_ln_1: LayerNorm,
    pub transformer_h_1_ln_2: LayerNorm,
    pub transformer_h_1_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_1_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_1_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_1_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_10_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_10_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_10_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_10_ln_1: LayerNorm,
    pub transformer_h_10_ln_2: LayerNorm,
    pub transformer_h_10_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_10_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_10_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_10_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_11_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_11_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_11_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_11_ln_1: LayerNorm,
    pub transformer_h_11_ln_2: LayerNorm,
    pub transformer_h_11_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_11_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_11_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_11_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_2_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_2_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_2_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_2_ln_1: LayerNorm,
    pub transformer_h_2_ln_2: LayerNorm,
    pub transformer_h_2_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_2_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_2_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_2_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_3_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_3_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_3_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_3_ln_1: LayerNorm,
    pub transformer_h_3_ln_2: LayerNorm,
    pub transformer_h_3_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_3_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_3_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_3_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_4_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_4_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_4_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_4_ln_1: LayerNorm,
    pub transformer_h_4_ln_2: LayerNorm,
    pub transformer_h_4_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_4_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_4_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_4_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_5_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_5_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_5_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_5_ln_1: LayerNorm,
    pub transformer_h_5_ln_2: LayerNorm,
    pub transformer_h_5_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_5_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_5_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_5_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_6_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_6_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_6_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_6_ln_1: LayerNorm,
    pub transformer_h_6_ln_2: LayerNorm,
    pub transformer_h_6_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_6_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_6_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_6_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_7_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_7_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_7_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_7_ln_1: LayerNorm,
    pub transformer_h_7_ln_2: LayerNorm,
    pub transformer_h_7_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_7_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_7_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_7_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_8_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_8_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_8_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_8_ln_1: LayerNorm,
    pub transformer_h_8_ln_2: LayerNorm,
    pub transformer_h_8_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_8_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_8_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_8_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_h_9_attn_c_attn: () /* TODO: Conv1D */,
    pub transformer_h_9_attn_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_9_attn_resid_dropout: () /* TODO: Dropout */,
    pub transformer_h_9_ln_1: LayerNorm,
    pub transformer_h_9_ln_2: LayerNorm,
    pub transformer_h_9_mlp_act: () /* TODO: NewGELUActivation */,
    pub transformer_h_9_mlp_c_fc: () /* TODO: Conv1D */,
    pub transformer_h_9_mlp_c_proj: () /* TODO: Conv1D */,
    pub transformer_h_9_mlp_dropout: () /* TODO: Dropout */,
    pub transformer_ln_f: LayerNorm,
    pub transformer_wpe: Embedding,
    pub transformer_wte: Embedding,
    pub checker: Option<PyChecker>,
}

impl GPT2Model {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let lm_head = candle_nn::linear_no_bias(768, 50257, vb.pp("lm_head"))?;
        let transformer_drop = todo!("Implement initialization for Dropout");
        let transformer_h_0_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_0_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_0_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_0_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.0.ln_1"))?;
        let transformer_h_0_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.0.ln_2"))?;
        let transformer_h_0_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_0_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_0_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_0_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_1_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_1_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_1_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_1_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.1.ln_1"))?;
        let transformer_h_1_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.1.ln_2"))?;
        let transformer_h_1_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_1_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_1_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_1_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_10_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_10_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_10_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_10_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.10.ln_1"))?;
        let transformer_h_10_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.10.ln_2"))?;
        let transformer_h_10_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_10_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_10_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_10_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_11_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_11_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_11_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_11_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.11.ln_1"))?;
        let transformer_h_11_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.11.ln_2"))?;
        let transformer_h_11_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_11_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_11_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_11_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_2_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_2_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_2_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_2_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.2.ln_1"))?;
        let transformer_h_2_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.2.ln_2"))?;
        let transformer_h_2_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_2_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_2_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_2_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_3_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_3_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_3_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_3_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.3.ln_1"))?;
        let transformer_h_3_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.3.ln_2"))?;
        let transformer_h_3_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_3_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_3_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_3_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_4_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_4_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_4_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_4_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.4.ln_1"))?;
        let transformer_h_4_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.4.ln_2"))?;
        let transformer_h_4_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_4_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_4_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_4_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_5_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_5_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_5_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_5_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.5.ln_1"))?;
        let transformer_h_5_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.5.ln_2"))?;
        let transformer_h_5_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_5_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_5_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_5_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_6_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_6_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_6_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_6_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.6.ln_1"))?;
        let transformer_h_6_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.6.ln_2"))?;
        let transformer_h_6_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_6_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_6_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_6_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_7_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_7_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_7_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_7_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.7.ln_1"))?;
        let transformer_h_7_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.7.ln_2"))?;
        let transformer_h_7_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_7_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_7_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_7_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_8_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_8_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_8_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_8_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.8.ln_1"))?;
        let transformer_h_8_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.8.ln_2"))?;
        let transformer_h_8_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_8_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_8_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_8_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_9_attn_c_attn = todo!("Implement initialization for Conv1D");
        let transformer_h_9_attn_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_9_attn_resid_dropout = todo!("Implement initialization for Dropout");
        let transformer_h_9_ln_1 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.9.ln_1"))?;
        let transformer_h_9_ln_2 = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.h.9.ln_2"))?;
        let transformer_h_9_mlp_act = todo!("Implement initialization for NewGELUActivation");
        let transformer_h_9_mlp_c_fc = todo!("Implement initialization for Conv1D");
        let transformer_h_9_mlp_c_proj = todo!("Implement initialization for Conv1D");
        let transformer_h_9_mlp_dropout = todo!("Implement initialization for Dropout");
        let transformer_ln_f = candle_nn::layer_norm(vec![768], candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("transformer.ln_f"))?;
        let transformer_wpe = candle_nn::embedding(1024, 768, vb.pp("transformer.wpe"))?;
        let transformer_wte = candle_nn::embedding(50257, 768, vb.pp("transformer.wte"))?;

        Ok(Self {
            lm_head,
            transformer_drop,
            transformer_h_0_attn_c_attn,
            transformer_h_0_attn_c_proj,
            transformer_h_0_attn_resid_dropout,
            transformer_h_0_ln_1,
            transformer_h_0_ln_2,
            transformer_h_0_mlp_act,
            transformer_h_0_mlp_c_fc,
            transformer_h_0_mlp_c_proj,
            transformer_h_0_mlp_dropout,
            transformer_h_1_attn_c_attn,
            transformer_h_1_attn_c_proj,
            transformer_h_1_attn_resid_dropout,
            transformer_h_1_ln_1,
            transformer_h_1_ln_2,
            transformer_h_1_mlp_act,
            transformer_h_1_mlp_c_fc,
            transformer_h_1_mlp_c_proj,
            transformer_h_1_mlp_dropout,
            transformer_h_10_attn_c_attn,
            transformer_h_10_attn_c_proj,
            transformer_h_10_attn_resid_dropout,
            transformer_h_10_ln_1,
            transformer_h_10_ln_2,
            transformer_h_10_mlp_act,
            transformer_h_10_mlp_c_fc,
            transformer_h_10_mlp_c_proj,
            transformer_h_10_mlp_dropout,
            transformer_h_11_attn_c_attn,
            transformer_h_11_attn_c_proj,
            transformer_h_11_attn_resid_dropout,
            transformer_h_11_ln_1,
            transformer_h_11_ln_2,
            transformer_h_11_mlp_act,
            transformer_h_11_mlp_c_fc,
            transformer_h_11_mlp_c_proj,
            transformer_h_11_mlp_dropout,
            transformer_h_2_attn_c_attn,
            transformer_h_2_attn_c_proj,
            transformer_h_2_attn_resid_dropout,
            transformer_h_2_ln_1,
            transformer_h_2_ln_2,
            transformer_h_2_mlp_act,
            transformer_h_2_mlp_c_fc,
            transformer_h_2_mlp_c_proj,
            transformer_h_2_mlp_dropout,
            transformer_h_3_attn_c_attn,
            transformer_h_3_attn_c_proj,
            transformer_h_3_attn_resid_dropout,
            transformer_h_3_ln_1,
            transformer_h_3_ln_2,
            transformer_h_3_mlp_act,
            transformer_h_3_mlp_c_fc,
            transformer_h_3_mlp_c_proj,
            transformer_h_3_mlp_dropout,
            transformer_h_4_attn_c_attn,
            transformer_h_4_attn_c_proj,
            transformer_h_4_attn_resid_dropout,
            transformer_h_4_ln_1,
            transformer_h_4_ln_2,
            transformer_h_4_mlp_act,
            transformer_h_4_mlp_c_fc,
            transformer_h_4_mlp_c_proj,
            transformer_h_4_mlp_dropout,
            transformer_h_5_attn_c_attn,
            transformer_h_5_attn_c_proj,
            transformer_h_5_attn_resid_dropout,
            transformer_h_5_ln_1,
            transformer_h_5_ln_2,
            transformer_h_5_mlp_act,
            transformer_h_5_mlp_c_fc,
            transformer_h_5_mlp_c_proj,
            transformer_h_5_mlp_dropout,
            transformer_h_6_attn_c_attn,
            transformer_h_6_attn_c_proj,
            transformer_h_6_attn_resid_dropout,
            transformer_h_6_ln_1,
            transformer_h_6_ln_2,
            transformer_h_6_mlp_act,
            transformer_h_6_mlp_c_fc,
            transformer_h_6_mlp_c_proj,
            transformer_h_6_mlp_dropout,
            transformer_h_7_attn_c_attn,
            transformer_h_7_attn_c_proj,
            transformer_h_7_attn_resid_dropout,
            transformer_h_7_ln_1,
            transformer_h_7_ln_2,
            transformer_h_7_mlp_act,
            transformer_h_7_mlp_c_fc,
            transformer_h_7_mlp_c_proj,
            transformer_h_7_mlp_dropout,
            transformer_h_8_attn_c_attn,
            transformer_h_8_attn_c_proj,
            transformer_h_8_attn_resid_dropout,
            transformer_h_8_ln_1,
            transformer_h_8_ln_2,
            transformer_h_8_mlp_act,
            transformer_h_8_mlp_c_fc,
            transformer_h_8_mlp_c_proj,
            transformer_h_8_mlp_dropout,
            transformer_h_9_attn_c_attn,
            transformer_h_9_attn_c_proj,
            transformer_h_9_attn_resid_dropout,
            transformer_h_9_ln_1,
            transformer_h_9_ln_2,
            transformer_h_9_mlp_act,
            transformer_h_9_mlp_c_fc,
            transformer_h_9_mlp_c_proj,
            transformer_h_9_mlp_dropout,
            transformer_ln_f,
            transformer_wpe,
            transformer_wte,
            checker,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: lm_head
        x = self.lm_head.forward(&x)?;
        py_check!(self.checker, "lm_head", &x);

        // Layer: transformer.drop
        x = self.transformer_drop.forward(&x)?;
        py_check!(self.checker, "transformer.drop", &x);

        // Layer: transformer.h.0.attn.c_attn
        x = self.transformer_h_0_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.attn.c_attn", &x);

        // Layer: transformer.h.0.attn.c_proj
        x = self.transformer_h_0_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.attn.c_proj", &x);

        // Layer: transformer.h.0.attn.resid_dropout
        x = self.transformer_h_0_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.attn.resid_dropout", &x);

        // Layer: transformer.h.0.ln_1
        x = self.transformer_h_0_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.ln_1", &x);

        // Layer: transformer.h.0.ln_2
        x = self.transformer_h_0_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.ln_2", &x);

        // Layer: transformer.h.0.mlp.act
        x = self.transformer_h_0_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.mlp.act", &x);

        // Layer: transformer.h.0.mlp.c_fc
        x = self.transformer_h_0_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.mlp.c_fc", &x);

        // Layer: transformer.h.0.mlp.c_proj
        x = self.transformer_h_0_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.mlp.c_proj", &x);

        // Layer: transformer.h.0.mlp.dropout
        x = self.transformer_h_0_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.0.mlp.dropout", &x);

        // Layer: transformer.h.1.attn.c_attn
        x = self.transformer_h_1_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.attn.c_attn", &x);

        // Layer: transformer.h.1.attn.c_proj
        x = self.transformer_h_1_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.attn.c_proj", &x);

        // Layer: transformer.h.1.attn.resid_dropout
        x = self.transformer_h_1_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.attn.resid_dropout", &x);

        // Layer: transformer.h.1.ln_1
        x = self.transformer_h_1_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.ln_1", &x);

        // Layer: transformer.h.1.ln_2
        x = self.transformer_h_1_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.ln_2", &x);

        // Layer: transformer.h.1.mlp.act
        x = self.transformer_h_1_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.mlp.act", &x);

        // Layer: transformer.h.1.mlp.c_fc
        x = self.transformer_h_1_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.mlp.c_fc", &x);

        // Layer: transformer.h.1.mlp.c_proj
        x = self.transformer_h_1_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.mlp.c_proj", &x);

        // Layer: transformer.h.1.mlp.dropout
        x = self.transformer_h_1_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.1.mlp.dropout", &x);

        // Layer: transformer.h.10.attn.c_attn
        x = self.transformer_h_10_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.attn.c_attn", &x);

        // Layer: transformer.h.10.attn.c_proj
        x = self.transformer_h_10_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.attn.c_proj", &x);

        // Layer: transformer.h.10.attn.resid_dropout
        x = self.transformer_h_10_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.attn.resid_dropout", &x);

        // Layer: transformer.h.10.ln_1
        x = self.transformer_h_10_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.ln_1", &x);

        // Layer: transformer.h.10.ln_2
        x = self.transformer_h_10_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.ln_2", &x);

        // Layer: transformer.h.10.mlp.act
        x = self.transformer_h_10_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.mlp.act", &x);

        // Layer: transformer.h.10.mlp.c_fc
        x = self.transformer_h_10_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.mlp.c_fc", &x);

        // Layer: transformer.h.10.mlp.c_proj
        x = self.transformer_h_10_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.mlp.c_proj", &x);

        // Layer: transformer.h.10.mlp.dropout
        x = self.transformer_h_10_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.10.mlp.dropout", &x);

        // Layer: transformer.h.11.attn.c_attn
        x = self.transformer_h_11_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.attn.c_attn", &x);

        // Layer: transformer.h.11.attn.c_proj
        x = self.transformer_h_11_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.attn.c_proj", &x);

        // Layer: transformer.h.11.attn.resid_dropout
        x = self.transformer_h_11_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.attn.resid_dropout", &x);

        // Layer: transformer.h.11.ln_1
        x = self.transformer_h_11_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.ln_1", &x);

        // Layer: transformer.h.11.ln_2
        x = self.transformer_h_11_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.ln_2", &x);

        // Layer: transformer.h.11.mlp.act
        x = self.transformer_h_11_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.mlp.act", &x);

        // Layer: transformer.h.11.mlp.c_fc
        x = self.transformer_h_11_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.mlp.c_fc", &x);

        // Layer: transformer.h.11.mlp.c_proj
        x = self.transformer_h_11_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.mlp.c_proj", &x);

        // Layer: transformer.h.11.mlp.dropout
        x = self.transformer_h_11_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.11.mlp.dropout", &x);

        // Layer: transformer.h.2.attn.c_attn
        x = self.transformer_h_2_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.attn.c_attn", &x);

        // Layer: transformer.h.2.attn.c_proj
        x = self.transformer_h_2_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.attn.c_proj", &x);

        // Layer: transformer.h.2.attn.resid_dropout
        x = self.transformer_h_2_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.attn.resid_dropout", &x);

        // Layer: transformer.h.2.ln_1
        x = self.transformer_h_2_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.ln_1", &x);

        // Layer: transformer.h.2.ln_2
        x = self.transformer_h_2_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.ln_2", &x);

        // Layer: transformer.h.2.mlp.act
        x = self.transformer_h_2_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.mlp.act", &x);

        // Layer: transformer.h.2.mlp.c_fc
        x = self.transformer_h_2_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.mlp.c_fc", &x);

        // Layer: transformer.h.2.mlp.c_proj
        x = self.transformer_h_2_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.mlp.c_proj", &x);

        // Layer: transformer.h.2.mlp.dropout
        x = self.transformer_h_2_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.2.mlp.dropout", &x);

        // Layer: transformer.h.3.attn.c_attn
        x = self.transformer_h_3_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.attn.c_attn", &x);

        // Layer: transformer.h.3.attn.c_proj
        x = self.transformer_h_3_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.attn.c_proj", &x);

        // Layer: transformer.h.3.attn.resid_dropout
        x = self.transformer_h_3_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.attn.resid_dropout", &x);

        // Layer: transformer.h.3.ln_1
        x = self.transformer_h_3_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.ln_1", &x);

        // Layer: transformer.h.3.ln_2
        x = self.transformer_h_3_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.ln_2", &x);

        // Layer: transformer.h.3.mlp.act
        x = self.transformer_h_3_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.mlp.act", &x);

        // Layer: transformer.h.3.mlp.c_fc
        x = self.transformer_h_3_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.mlp.c_fc", &x);

        // Layer: transformer.h.3.mlp.c_proj
        x = self.transformer_h_3_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.mlp.c_proj", &x);

        // Layer: transformer.h.3.mlp.dropout
        x = self.transformer_h_3_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.3.mlp.dropout", &x);

        // Layer: transformer.h.4.attn.c_attn
        x = self.transformer_h_4_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.attn.c_attn", &x);

        // Layer: transformer.h.4.attn.c_proj
        x = self.transformer_h_4_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.attn.c_proj", &x);

        // Layer: transformer.h.4.attn.resid_dropout
        x = self.transformer_h_4_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.attn.resid_dropout", &x);

        // Layer: transformer.h.4.ln_1
        x = self.transformer_h_4_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.ln_1", &x);

        // Layer: transformer.h.4.ln_2
        x = self.transformer_h_4_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.ln_2", &x);

        // Layer: transformer.h.4.mlp.act
        x = self.transformer_h_4_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.mlp.act", &x);

        // Layer: transformer.h.4.mlp.c_fc
        x = self.transformer_h_4_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.mlp.c_fc", &x);

        // Layer: transformer.h.4.mlp.c_proj
        x = self.transformer_h_4_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.mlp.c_proj", &x);

        // Layer: transformer.h.4.mlp.dropout
        x = self.transformer_h_4_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.4.mlp.dropout", &x);

        // Layer: transformer.h.5.attn.c_attn
        x = self.transformer_h_5_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.attn.c_attn", &x);

        // Layer: transformer.h.5.attn.c_proj
        x = self.transformer_h_5_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.attn.c_proj", &x);

        // Layer: transformer.h.5.attn.resid_dropout
        x = self.transformer_h_5_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.attn.resid_dropout", &x);

        // Layer: transformer.h.5.ln_1
        x = self.transformer_h_5_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.ln_1", &x);

        // Layer: transformer.h.5.ln_2
        x = self.transformer_h_5_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.ln_2", &x);

        // Layer: transformer.h.5.mlp.act
        x = self.transformer_h_5_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.mlp.act", &x);

        // Layer: transformer.h.5.mlp.c_fc
        x = self.transformer_h_5_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.mlp.c_fc", &x);

        // Layer: transformer.h.5.mlp.c_proj
        x = self.transformer_h_5_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.mlp.c_proj", &x);

        // Layer: transformer.h.5.mlp.dropout
        x = self.transformer_h_5_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.5.mlp.dropout", &x);

        // Layer: transformer.h.6.attn.c_attn
        x = self.transformer_h_6_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.attn.c_attn", &x);

        // Layer: transformer.h.6.attn.c_proj
        x = self.transformer_h_6_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.attn.c_proj", &x);

        // Layer: transformer.h.6.attn.resid_dropout
        x = self.transformer_h_6_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.attn.resid_dropout", &x);

        // Layer: transformer.h.6.ln_1
        x = self.transformer_h_6_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.ln_1", &x);

        // Layer: transformer.h.6.ln_2
        x = self.transformer_h_6_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.ln_2", &x);

        // Layer: transformer.h.6.mlp.act
        x = self.transformer_h_6_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.mlp.act", &x);

        // Layer: transformer.h.6.mlp.c_fc
        x = self.transformer_h_6_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.mlp.c_fc", &x);

        // Layer: transformer.h.6.mlp.c_proj
        x = self.transformer_h_6_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.mlp.c_proj", &x);

        // Layer: transformer.h.6.mlp.dropout
        x = self.transformer_h_6_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.6.mlp.dropout", &x);

        // Layer: transformer.h.7.attn.c_attn
        x = self.transformer_h_7_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.attn.c_attn", &x);

        // Layer: transformer.h.7.attn.c_proj
        x = self.transformer_h_7_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.attn.c_proj", &x);

        // Layer: transformer.h.7.attn.resid_dropout
        x = self.transformer_h_7_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.attn.resid_dropout", &x);

        // Layer: transformer.h.7.ln_1
        x = self.transformer_h_7_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.ln_1", &x);

        // Layer: transformer.h.7.ln_2
        x = self.transformer_h_7_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.ln_2", &x);

        // Layer: transformer.h.7.mlp.act
        x = self.transformer_h_7_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.mlp.act", &x);

        // Layer: transformer.h.7.mlp.c_fc
        x = self.transformer_h_7_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.mlp.c_fc", &x);

        // Layer: transformer.h.7.mlp.c_proj
        x = self.transformer_h_7_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.mlp.c_proj", &x);

        // Layer: transformer.h.7.mlp.dropout
        x = self.transformer_h_7_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.7.mlp.dropout", &x);

        // Layer: transformer.h.8.attn.c_attn
        x = self.transformer_h_8_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.attn.c_attn", &x);

        // Layer: transformer.h.8.attn.c_proj
        x = self.transformer_h_8_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.attn.c_proj", &x);

        // Layer: transformer.h.8.attn.resid_dropout
        x = self.transformer_h_8_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.attn.resid_dropout", &x);

        // Layer: transformer.h.8.ln_1
        x = self.transformer_h_8_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.ln_1", &x);

        // Layer: transformer.h.8.ln_2
        x = self.transformer_h_8_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.ln_2", &x);

        // Layer: transformer.h.8.mlp.act
        x = self.transformer_h_8_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.mlp.act", &x);

        // Layer: transformer.h.8.mlp.c_fc
        x = self.transformer_h_8_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.mlp.c_fc", &x);

        // Layer: transformer.h.8.mlp.c_proj
        x = self.transformer_h_8_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.mlp.c_proj", &x);

        // Layer: transformer.h.8.mlp.dropout
        x = self.transformer_h_8_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.8.mlp.dropout", &x);

        // Layer: transformer.h.9.attn.c_attn
        x = self.transformer_h_9_attn_c_attn.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.attn.c_attn", &x);

        // Layer: transformer.h.9.attn.c_proj
        x = self.transformer_h_9_attn_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.attn.c_proj", &x);

        // Layer: transformer.h.9.attn.resid_dropout
        x = self.transformer_h_9_attn_resid_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.attn.resid_dropout", &x);

        // Layer: transformer.h.9.ln_1
        x = self.transformer_h_9_ln_1.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.ln_1", &x);

        // Layer: transformer.h.9.ln_2
        x = self.transformer_h_9_ln_2.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.ln_2", &x);

        // Layer: transformer.h.9.mlp.act
        x = self.transformer_h_9_mlp_act.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.mlp.act", &x);

        // Layer: transformer.h.9.mlp.c_fc
        x = self.transformer_h_9_mlp_c_fc.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.mlp.c_fc", &x);

        // Layer: transformer.h.9.mlp.c_proj
        x = self.transformer_h_9_mlp_c_proj.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.mlp.c_proj", &x);

        // Layer: transformer.h.9.mlp.dropout
        x = self.transformer_h_9_mlp_dropout.forward(&x)?;
        py_check!(self.checker, "transformer.h.9.mlp.dropout", &x);

        // Layer: transformer.ln_f
        x = self.transformer_ln_f.forward(&x)?;
        py_check!(self.checker, "transformer.ln_f", &x);

        // Layer: transformer.wpe
        x = self.transformer_wpe.forward(&x)?;
        py_check!(self.checker, "transformer.wpe", &x);

        // Layer: transformer.wte
        x = self.transformer_wte.forward(&x)?;
        py_check!(self.checker, "transformer.wte", &x);

        Ok(x)
    }
}
