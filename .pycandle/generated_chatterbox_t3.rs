use candle_core::{Result, Tensor, IndexOp, Shape};
use candle_nn::{Module, VarBuilder};
use pycandle_core::{PyChecker, py_check, VerificationMode, layers::*};

#[derive(Debug, Clone, Copy, Default)]
pub struct Config {
    pub context_length: usize, // 8196
    pub hidden_dim: usize, // 1024
    pub n_head: usize, // 16
    pub n_layers: usize, // 24
    pub vocab_size: usize, // 50276
}
pub struct ChatterboxT3 {
    pub drop: Dropout,
    pub h_0_attn_c_attn: candle_nn::Linear,
    pub h_0_attn_c_proj: candle_nn::Linear,
    pub h_0_attn_resid_dropout: Dropout,
    pub h_0_ln_1: candle_nn::LayerNorm,
    pub h_0_ln_2: candle_nn::LayerNorm,
    pub h_0_mlp_act: candle_nn::Activation,
    pub h_0_mlp_c_fc: candle_nn::Linear,
    pub h_0_mlp_c_proj: candle_nn::Linear,
    pub h_0_mlp_dropout: Dropout,
    pub h_1_attn_c_attn: candle_nn::Linear,
    pub h_1_attn_c_proj: candle_nn::Linear,
    pub h_1_attn_resid_dropout: Dropout,
    pub h_1_ln_1: candle_nn::LayerNorm,
    pub h_1_ln_2: candle_nn::LayerNorm,
    pub h_1_mlp_act: candle_nn::Activation,
    pub h_1_mlp_c_fc: candle_nn::Linear,
    pub h_1_mlp_c_proj: candle_nn::Linear,
    pub h_1_mlp_dropout: Dropout,
    pub h_10_attn_c_attn: candle_nn::Linear,
    pub h_10_attn_c_proj: candle_nn::Linear,
    pub h_10_attn_resid_dropout: Dropout,
    pub h_10_ln_1: candle_nn::LayerNorm,
    pub h_10_ln_2: candle_nn::LayerNorm,
    pub h_10_mlp_act: candle_nn::Activation,
    pub h_10_mlp_c_fc: candle_nn::Linear,
    pub h_10_mlp_c_proj: candle_nn::Linear,
    pub h_10_mlp_dropout: Dropout,
    pub h_11_attn_c_attn: candle_nn::Linear,
    pub h_11_attn_c_proj: candle_nn::Linear,
    pub h_11_attn_resid_dropout: Dropout,
    pub h_11_ln_1: candle_nn::LayerNorm,
    pub h_11_ln_2: candle_nn::LayerNorm,
    pub h_11_mlp_act: candle_nn::Activation,
    pub h_11_mlp_c_fc: candle_nn::Linear,
    pub h_11_mlp_c_proj: candle_nn::Linear,
    pub h_11_mlp_dropout: Dropout,
    pub h_12_attn_c_attn: candle_nn::Linear,
    pub h_12_attn_c_proj: candle_nn::Linear,
    pub h_12_attn_resid_dropout: Dropout,
    pub h_12_ln_1: candle_nn::LayerNorm,
    pub h_12_ln_2: candle_nn::LayerNorm,
    pub h_12_mlp_act: candle_nn::Activation,
    pub h_12_mlp_c_fc: candle_nn::Linear,
    pub h_12_mlp_c_proj: candle_nn::Linear,
    pub h_12_mlp_dropout: Dropout,
    pub h_13_attn_c_attn: candle_nn::Linear,
    pub h_13_attn_c_proj: candle_nn::Linear,
    pub h_13_attn_resid_dropout: Dropout,
    pub h_13_ln_1: candle_nn::LayerNorm,
    pub h_13_ln_2: candle_nn::LayerNorm,
    pub h_13_mlp_act: candle_nn::Activation,
    pub h_13_mlp_c_fc: candle_nn::Linear,
    pub h_13_mlp_c_proj: candle_nn::Linear,
    pub h_13_mlp_dropout: Dropout,
    pub h_14_attn_c_attn: candle_nn::Linear,
    pub h_14_attn_c_proj: candle_nn::Linear,
    pub h_14_attn_resid_dropout: Dropout,
    pub h_14_ln_1: candle_nn::LayerNorm,
    pub h_14_ln_2: candle_nn::LayerNorm,
    pub h_14_mlp_act: candle_nn::Activation,
    pub h_14_mlp_c_fc: candle_nn::Linear,
    pub h_14_mlp_c_proj: candle_nn::Linear,
    pub h_14_mlp_dropout: Dropout,
    pub h_15_attn_c_attn: candle_nn::Linear,
    pub h_15_attn_c_proj: candle_nn::Linear,
    pub h_15_attn_resid_dropout: Dropout,
    pub h_15_ln_1: candle_nn::LayerNorm,
    pub h_15_ln_2: candle_nn::LayerNorm,
    pub h_15_mlp_act: candle_nn::Activation,
    pub h_15_mlp_c_fc: candle_nn::Linear,
    pub h_15_mlp_c_proj: candle_nn::Linear,
    pub h_15_mlp_dropout: Dropout,
    pub h_16_attn_c_attn: candle_nn::Linear,
    pub h_16_attn_c_proj: candle_nn::Linear,
    pub h_16_attn_resid_dropout: Dropout,
    pub h_16_ln_1: candle_nn::LayerNorm,
    pub h_16_ln_2: candle_nn::LayerNorm,
    pub h_16_mlp_act: candle_nn::Activation,
    pub h_16_mlp_c_fc: candle_nn::Linear,
    pub h_16_mlp_c_proj: candle_nn::Linear,
    pub h_16_mlp_dropout: Dropout,
    pub h_17_attn_c_attn: candle_nn::Linear,
    pub h_17_attn_c_proj: candle_nn::Linear,
    pub h_17_attn_resid_dropout: Dropout,
    pub h_17_ln_1: candle_nn::LayerNorm,
    pub h_17_ln_2: candle_nn::LayerNorm,
    pub h_17_mlp_act: candle_nn::Activation,
    pub h_17_mlp_c_fc: candle_nn::Linear,
    pub h_17_mlp_c_proj: candle_nn::Linear,
    pub h_17_mlp_dropout: Dropout,
    pub h_18_attn_c_attn: candle_nn::Linear,
    pub h_18_attn_c_proj: candle_nn::Linear,
    pub h_18_attn_resid_dropout: Dropout,
    pub h_18_ln_1: candle_nn::LayerNorm,
    pub h_18_ln_2: candle_nn::LayerNorm,
    pub h_18_mlp_act: candle_nn::Activation,
    pub h_18_mlp_c_fc: candle_nn::Linear,
    pub h_18_mlp_c_proj: candle_nn::Linear,
    pub h_18_mlp_dropout: Dropout,
    pub h_19_attn_c_attn: candle_nn::Linear,
    pub h_19_attn_c_proj: candle_nn::Linear,
    pub h_19_attn_resid_dropout: Dropout,
    pub h_19_ln_1: candle_nn::LayerNorm,
    pub h_19_ln_2: candle_nn::LayerNorm,
    pub h_19_mlp_act: candle_nn::Activation,
    pub h_19_mlp_c_fc: candle_nn::Linear,
    pub h_19_mlp_c_proj: candle_nn::Linear,
    pub h_19_mlp_dropout: Dropout,
    pub h_2_attn_c_attn: candle_nn::Linear,
    pub h_2_attn_c_proj: candle_nn::Linear,
    pub h_2_attn_resid_dropout: Dropout,
    pub h_2_ln_1: candle_nn::LayerNorm,
    pub h_2_ln_2: candle_nn::LayerNorm,
    pub h_2_mlp_act: candle_nn::Activation,
    pub h_2_mlp_c_fc: candle_nn::Linear,
    pub h_2_mlp_c_proj: candle_nn::Linear,
    pub h_2_mlp_dropout: Dropout,
    pub h_20_attn_c_attn: candle_nn::Linear,
    pub h_20_attn_c_proj: candle_nn::Linear,
    pub h_20_attn_resid_dropout: Dropout,
    pub h_20_ln_1: candle_nn::LayerNorm,
    pub h_20_ln_2: candle_nn::LayerNorm,
    pub h_20_mlp_act: candle_nn::Activation,
    pub h_20_mlp_c_fc: candle_nn::Linear,
    pub h_20_mlp_c_proj: candle_nn::Linear,
    pub h_20_mlp_dropout: Dropout,
    pub h_21_attn_c_attn: candle_nn::Linear,
    pub h_21_attn_c_proj: candle_nn::Linear,
    pub h_21_attn_resid_dropout: Dropout,
    pub h_21_ln_1: candle_nn::LayerNorm,
    pub h_21_ln_2: candle_nn::LayerNorm,
    pub h_21_mlp_act: candle_nn::Activation,
    pub h_21_mlp_c_fc: candle_nn::Linear,
    pub h_21_mlp_c_proj: candle_nn::Linear,
    pub h_21_mlp_dropout: Dropout,
    pub h_22_attn_c_attn: candle_nn::Linear,
    pub h_22_attn_c_proj: candle_nn::Linear,
    pub h_22_attn_resid_dropout: Dropout,
    pub h_22_ln_1: candle_nn::LayerNorm,
    pub h_22_ln_2: candle_nn::LayerNorm,
    pub h_22_mlp_act: candle_nn::Activation,
    pub h_22_mlp_c_fc: candle_nn::Linear,
    pub h_22_mlp_c_proj: candle_nn::Linear,
    pub h_22_mlp_dropout: Dropout,
    pub h_23_attn_c_attn: candle_nn::Linear,
    pub h_23_attn_c_proj: candle_nn::Linear,
    pub h_23_attn_resid_dropout: Dropout,
    pub h_23_ln_1: candle_nn::LayerNorm,
    pub h_23_ln_2: candle_nn::LayerNorm,
    pub h_23_mlp_act: candle_nn::Activation,
    pub h_23_mlp_c_fc: candle_nn::Linear,
    pub h_23_mlp_c_proj: candle_nn::Linear,
    pub h_23_mlp_dropout: Dropout,
    pub h_3_attn_c_attn: candle_nn::Linear,
    pub h_3_attn_c_proj: candle_nn::Linear,
    pub h_3_attn_resid_dropout: Dropout,
    pub h_3_ln_1: candle_nn::LayerNorm,
    pub h_3_ln_2: candle_nn::LayerNorm,
    pub h_3_mlp_act: candle_nn::Activation,
    pub h_3_mlp_c_fc: candle_nn::Linear,
    pub h_3_mlp_c_proj: candle_nn::Linear,
    pub h_3_mlp_dropout: Dropout,
    pub h_4_attn_c_attn: candle_nn::Linear,
    pub h_4_attn_c_proj: candle_nn::Linear,
    pub h_4_attn_resid_dropout: Dropout,
    pub h_4_ln_1: candle_nn::LayerNorm,
    pub h_4_ln_2: candle_nn::LayerNorm,
    pub h_4_mlp_act: candle_nn::Activation,
    pub h_4_mlp_c_fc: candle_nn::Linear,
    pub h_4_mlp_c_proj: candle_nn::Linear,
    pub h_4_mlp_dropout: Dropout,
    pub h_5_attn_c_attn: candle_nn::Linear,
    pub h_5_attn_c_proj: candle_nn::Linear,
    pub h_5_attn_resid_dropout: Dropout,
    pub h_5_ln_1: candle_nn::LayerNorm,
    pub h_5_ln_2: candle_nn::LayerNorm,
    pub h_5_mlp_act: candle_nn::Activation,
    pub h_5_mlp_c_fc: candle_nn::Linear,
    pub h_5_mlp_c_proj: candle_nn::Linear,
    pub h_5_mlp_dropout: Dropout,
    pub h_6_attn_c_attn: candle_nn::Linear,
    pub h_6_attn_c_proj: candle_nn::Linear,
    pub h_6_attn_resid_dropout: Dropout,
    pub h_6_ln_1: candle_nn::LayerNorm,
    pub h_6_ln_2: candle_nn::LayerNorm,
    pub h_6_mlp_act: candle_nn::Activation,
    pub h_6_mlp_c_fc: candle_nn::Linear,
    pub h_6_mlp_c_proj: candle_nn::Linear,
    pub h_6_mlp_dropout: Dropout,
    pub h_7_attn_c_attn: candle_nn::Linear,
    pub h_7_attn_c_proj: candle_nn::Linear,
    pub h_7_attn_resid_dropout: Dropout,
    pub h_7_ln_1: candle_nn::LayerNorm,
    pub h_7_ln_2: candle_nn::LayerNorm,
    pub h_7_mlp_act: candle_nn::Activation,
    pub h_7_mlp_c_fc: candle_nn::Linear,
    pub h_7_mlp_c_proj: candle_nn::Linear,
    pub h_7_mlp_dropout: Dropout,
    pub h_8_attn_c_attn: candle_nn::Linear,
    pub h_8_attn_c_proj: candle_nn::Linear,
    pub h_8_attn_resid_dropout: Dropout,
    pub h_8_ln_1: candle_nn::LayerNorm,
    pub h_8_ln_2: candle_nn::LayerNorm,
    pub h_8_mlp_act: candle_nn::Activation,
    pub h_8_mlp_c_fc: candle_nn::Linear,
    pub h_8_mlp_c_proj: candle_nn::Linear,
    pub h_8_mlp_dropout: Dropout,
    pub h_9_attn_c_attn: candle_nn::Linear,
    pub h_9_attn_c_proj: candle_nn::Linear,
    pub h_9_attn_resid_dropout: Dropout,
    pub h_9_ln_1: candle_nn::LayerNorm,
    pub h_9_ln_2: candle_nn::LayerNorm,
    pub h_9_mlp_act: candle_nn::Activation,
    pub h_9_mlp_c_fc: candle_nn::Linear,
    pub h_9_mlp_c_proj: candle_nn::Linear,
    pub h_9_mlp_dropout: Dropout,
    pub ln_f: candle_nn::LayerNorm,
    pub wpe: candle_nn::Embedding,
    pub wte: candle_nn::Embedding,
    pub checker: Option<PyChecker>,
}

impl ChatterboxT3 {
    #[allow(unused_variables)]
    pub fn load(config: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let gpt2_cfg = pycandle_core::gpt2::Config {
            vocab_size: config.vocab_size,
            context_length: config.context_length,
            emb_dim: config.hidden_dim,
            n_heads: config.n_head,
            n_layers: config.n_layers,
            ..Default::default()
        };
        let drop = Dropout::new();
        let h_0_attn_c_attn = { let w = vb.pp("h.0.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.0.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_0_attn_c_proj = { let w = vb.pp("h.0.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.0.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_0_attn_resid_dropout = Dropout::new();
        let h_0_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.0.ln_1"))?;
        let h_0_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.0.ln_2"))?;
        let h_0_mlp_act = candle_nn::Activation::NewGelu;
        let h_0_mlp_c_fc = { let w = vb.pp("h.0.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.0.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_0_mlp_c_proj = { let w = vb.pp("h.0.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.0.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_0_mlp_dropout = Dropout::new();
        let h_1_attn_c_attn = { let w = vb.pp("h.1.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.1.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_1_attn_c_proj = { let w = vb.pp("h.1.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.1.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_1_attn_resid_dropout = Dropout::new();
        let h_1_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.1.ln_1"))?;
        let h_1_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.1.ln_2"))?;
        let h_1_mlp_act = candle_nn::Activation::NewGelu;
        let h_1_mlp_c_fc = { let w = vb.pp("h.1.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.1.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_1_mlp_c_proj = { let w = vb.pp("h.1.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.1.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_1_mlp_dropout = Dropout::new();
        let h_10_attn_c_attn = { let w = vb.pp("h.10.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.10.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_10_attn_c_proj = { let w = vb.pp("h.10.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.10.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_10_attn_resid_dropout = Dropout::new();
        let h_10_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.10.ln_1"))?;
        let h_10_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.10.ln_2"))?;
        let h_10_mlp_act = candle_nn::Activation::NewGelu;
        let h_10_mlp_c_fc = { let w = vb.pp("h.10.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.10.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_10_mlp_c_proj = { let w = vb.pp("h.10.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.10.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_10_mlp_dropout = Dropout::new();
        let h_11_attn_c_attn = { let w = vb.pp("h.11.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.11.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_11_attn_c_proj = { let w = vb.pp("h.11.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.11.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_11_attn_resid_dropout = Dropout::new();
        let h_11_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.11.ln_1"))?;
        let h_11_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.11.ln_2"))?;
        let h_11_mlp_act = candle_nn::Activation::NewGelu;
        let h_11_mlp_c_fc = { let w = vb.pp("h.11.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.11.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_11_mlp_c_proj = { let w = vb.pp("h.11.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.11.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_11_mlp_dropout = Dropout::new();
        let h_12_attn_c_attn = { let w = vb.pp("h.12.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.12.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_12_attn_c_proj = { let w = vb.pp("h.12.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.12.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_12_attn_resid_dropout = Dropout::new();
        let h_12_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.12.ln_1"))?;
        let h_12_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.12.ln_2"))?;
        let h_12_mlp_act = candle_nn::Activation::NewGelu;
        let h_12_mlp_c_fc = { let w = vb.pp("h.12.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.12.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_12_mlp_c_proj = { let w = vb.pp("h.12.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.12.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_12_mlp_dropout = Dropout::new();
        let h_13_attn_c_attn = { let w = vb.pp("h.13.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.13.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_13_attn_c_proj = { let w = vb.pp("h.13.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.13.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_13_attn_resid_dropout = Dropout::new();
        let h_13_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.13.ln_1"))?;
        let h_13_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.13.ln_2"))?;
        let h_13_mlp_act = candle_nn::Activation::NewGelu;
        let h_13_mlp_c_fc = { let w = vb.pp("h.13.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.13.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_13_mlp_c_proj = { let w = vb.pp("h.13.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.13.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_13_mlp_dropout = Dropout::new();
        let h_14_attn_c_attn = { let w = vb.pp("h.14.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.14.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_14_attn_c_proj = { let w = vb.pp("h.14.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.14.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_14_attn_resid_dropout = Dropout::new();
        let h_14_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.14.ln_1"))?;
        let h_14_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.14.ln_2"))?;
        let h_14_mlp_act = candle_nn::Activation::NewGelu;
        let h_14_mlp_c_fc = { let w = vb.pp("h.14.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.14.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_14_mlp_c_proj = { let w = vb.pp("h.14.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.14.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_14_mlp_dropout = Dropout::new();
        let h_15_attn_c_attn = { let w = vb.pp("h.15.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.15.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_15_attn_c_proj = { let w = vb.pp("h.15.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.15.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_15_attn_resid_dropout = Dropout::new();
        let h_15_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.15.ln_1"))?;
        let h_15_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.15.ln_2"))?;
        let h_15_mlp_act = candle_nn::Activation::NewGelu;
        let h_15_mlp_c_fc = { let w = vb.pp("h.15.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.15.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_15_mlp_c_proj = { let w = vb.pp("h.15.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.15.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_15_mlp_dropout = Dropout::new();
        let h_16_attn_c_attn = { let w = vb.pp("h.16.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.16.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_16_attn_c_proj = { let w = vb.pp("h.16.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.16.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_16_attn_resid_dropout = Dropout::new();
        let h_16_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.16.ln_1"))?;
        let h_16_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.16.ln_2"))?;
        let h_16_mlp_act = candle_nn::Activation::NewGelu;
        let h_16_mlp_c_fc = { let w = vb.pp("h.16.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.16.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_16_mlp_c_proj = { let w = vb.pp("h.16.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.16.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_16_mlp_dropout = Dropout::new();
        let h_17_attn_c_attn = { let w = vb.pp("h.17.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.17.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_17_attn_c_proj = { let w = vb.pp("h.17.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.17.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_17_attn_resid_dropout = Dropout::new();
        let h_17_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.17.ln_1"))?;
        let h_17_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.17.ln_2"))?;
        let h_17_mlp_act = candle_nn::Activation::NewGelu;
        let h_17_mlp_c_fc = { let w = vb.pp("h.17.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.17.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_17_mlp_c_proj = { let w = vb.pp("h.17.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.17.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_17_mlp_dropout = Dropout::new();
        let h_18_attn_c_attn = { let w = vb.pp("h.18.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.18.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_18_attn_c_proj = { let w = vb.pp("h.18.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.18.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_18_attn_resid_dropout = Dropout::new();
        let h_18_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.18.ln_1"))?;
        let h_18_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.18.ln_2"))?;
        let h_18_mlp_act = candle_nn::Activation::NewGelu;
        let h_18_mlp_c_fc = { let w = vb.pp("h.18.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.18.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_18_mlp_c_proj = { let w = vb.pp("h.18.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.18.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_18_mlp_dropout = Dropout::new();
        let h_19_attn_c_attn = { let w = vb.pp("h.19.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.19.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_19_attn_c_proj = { let w = vb.pp("h.19.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.19.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_19_attn_resid_dropout = Dropout::new();
        let h_19_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.19.ln_1"))?;
        let h_19_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.19.ln_2"))?;
        let h_19_mlp_act = candle_nn::Activation::NewGelu;
        let h_19_mlp_c_fc = { let w = vb.pp("h.19.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.19.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_19_mlp_c_proj = { let w = vb.pp("h.19.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.19.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_19_mlp_dropout = Dropout::new();
        let h_2_attn_c_attn = { let w = vb.pp("h.2.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.2.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_2_attn_c_proj = { let w = vb.pp("h.2.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.2.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_2_attn_resid_dropout = Dropout::new();
        let h_2_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.2.ln_1"))?;
        let h_2_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.2.ln_2"))?;
        let h_2_mlp_act = candle_nn::Activation::NewGelu;
        let h_2_mlp_c_fc = { let w = vb.pp("h.2.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.2.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_2_mlp_c_proj = { let w = vb.pp("h.2.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.2.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_2_mlp_dropout = Dropout::new();
        let h_20_attn_c_attn = { let w = vb.pp("h.20.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.20.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_20_attn_c_proj = { let w = vb.pp("h.20.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.20.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_20_attn_resid_dropout = Dropout::new();
        let h_20_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.20.ln_1"))?;
        let h_20_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.20.ln_2"))?;
        let h_20_mlp_act = candle_nn::Activation::NewGelu;
        let h_20_mlp_c_fc = { let w = vb.pp("h.20.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.20.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_20_mlp_c_proj = { let w = vb.pp("h.20.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.20.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_20_mlp_dropout = Dropout::new();
        let h_21_attn_c_attn = { let w = vb.pp("h.21.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.21.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_21_attn_c_proj = { let w = vb.pp("h.21.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.21.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_21_attn_resid_dropout = Dropout::new();
        let h_21_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.21.ln_1"))?;
        let h_21_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.21.ln_2"))?;
        let h_21_mlp_act = candle_nn::Activation::NewGelu;
        let h_21_mlp_c_fc = { let w = vb.pp("h.21.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.21.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_21_mlp_c_proj = { let w = vb.pp("h.21.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.21.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_21_mlp_dropout = Dropout::new();
        let h_22_attn_c_attn = { let w = vb.pp("h.22.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.22.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_22_attn_c_proj = { let w = vb.pp("h.22.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.22.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_22_attn_resid_dropout = Dropout::new();
        let h_22_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.22.ln_1"))?;
        let h_22_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.22.ln_2"))?;
        let h_22_mlp_act = candle_nn::Activation::NewGelu;
        let h_22_mlp_c_fc = { let w = vb.pp("h.22.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.22.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_22_mlp_c_proj = { let w = vb.pp("h.22.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.22.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_22_mlp_dropout = Dropout::new();
        let h_23_attn_c_attn = { let w = vb.pp("h.23.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.23.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_23_attn_c_proj = { let w = vb.pp("h.23.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.23.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_23_attn_resid_dropout = Dropout::new();
        let h_23_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.23.ln_1"))?;
        let h_23_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.23.ln_2"))?;
        let h_23_mlp_act = candle_nn::Activation::NewGelu;
        let h_23_mlp_c_fc = { let w = vb.pp("h.23.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.23.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_23_mlp_c_proj = { let w = vb.pp("h.23.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.23.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_23_mlp_dropout = Dropout::new();
        let h_3_attn_c_attn = { let w = vb.pp("h.3.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.3.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_3_attn_c_proj = { let w = vb.pp("h.3.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.3.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_3_attn_resid_dropout = Dropout::new();
        let h_3_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.3.ln_1"))?;
        let h_3_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.3.ln_2"))?;
        let h_3_mlp_act = candle_nn::Activation::NewGelu;
        let h_3_mlp_c_fc = { let w = vb.pp("h.3.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.3.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_3_mlp_c_proj = { let w = vb.pp("h.3.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.3.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_3_mlp_dropout = Dropout::new();
        let h_4_attn_c_attn = { let w = vb.pp("h.4.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.4.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_4_attn_c_proj = { let w = vb.pp("h.4.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.4.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_4_attn_resid_dropout = Dropout::new();
        let h_4_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.4.ln_1"))?;
        let h_4_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.4.ln_2"))?;
        let h_4_mlp_act = candle_nn::Activation::NewGelu;
        let h_4_mlp_c_fc = { let w = vb.pp("h.4.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.4.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_4_mlp_c_proj = { let w = vb.pp("h.4.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.4.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_4_mlp_dropout = Dropout::new();
        let h_5_attn_c_attn = { let w = vb.pp("h.5.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.5.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_5_attn_c_proj = { let w = vb.pp("h.5.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.5.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_5_attn_resid_dropout = Dropout::new();
        let h_5_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.5.ln_1"))?;
        let h_5_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.5.ln_2"))?;
        let h_5_mlp_act = candle_nn::Activation::NewGelu;
        let h_5_mlp_c_fc = { let w = vb.pp("h.5.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.5.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_5_mlp_c_proj = { let w = vb.pp("h.5.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.5.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_5_mlp_dropout = Dropout::new();
        let h_6_attn_c_attn = { let w = vb.pp("h.6.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.6.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_6_attn_c_proj = { let w = vb.pp("h.6.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.6.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_6_attn_resid_dropout = Dropout::new();
        let h_6_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.6.ln_1"))?;
        let h_6_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.6.ln_2"))?;
        let h_6_mlp_act = candle_nn::Activation::NewGelu;
        let h_6_mlp_c_fc = { let w = vb.pp("h.6.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.6.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_6_mlp_c_proj = { let w = vb.pp("h.6.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.6.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_6_mlp_dropout = Dropout::new();
        let h_7_attn_c_attn = { let w = vb.pp("h.7.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.7.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_7_attn_c_proj = { let w = vb.pp("h.7.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.7.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_7_attn_resid_dropout = Dropout::new();
        let h_7_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.7.ln_1"))?;
        let h_7_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.7.ln_2"))?;
        let h_7_mlp_act = candle_nn::Activation::NewGelu;
        let h_7_mlp_c_fc = { let w = vb.pp("h.7.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.7.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_7_mlp_c_proj = { let w = vb.pp("h.7.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.7.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_7_mlp_dropout = Dropout::new();
        let h_8_attn_c_attn = { let w = vb.pp("h.8.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.8.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_8_attn_c_proj = { let w = vb.pp("h.8.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.8.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_8_attn_resid_dropout = Dropout::new();
        let h_8_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.8.ln_1"))?;
        let h_8_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.8.ln_2"))?;
        let h_8_mlp_act = candle_nn::Activation::NewGelu;
        let h_8_mlp_c_fc = { let w = vb.pp("h.8.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.8.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_8_mlp_c_proj = { let w = vb.pp("h.8.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.8.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_8_mlp_dropout = Dropout::new();
        let h_9_attn_c_attn = { let w = vb.pp("h.9.attn.c_attn").get((config.hidden_dim, 3072), "weight")?.t()?; let b = Some(vb.pp("h.9.attn.c_attn").get(3072, "bias")?); candle_nn::Linear::new(w, b) };
        let h_9_attn_c_proj = { let w = vb.pp("h.9.attn.c_proj").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.9.attn.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_9_attn_resid_dropout = Dropout::new();
        let h_9_ln_1 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.9.ln_1"))?;
        let h_9_ln_2 = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("h.9.ln_2"))?;
        let h_9_mlp_act = candle_nn::Activation::NewGelu;
        let h_9_mlp_c_fc = { let w = vb.pp("h.9.mlp.c_fc").get((config.hidden_dim, 4096), "weight")?.t()?; let b = Some(vb.pp("h.9.mlp.c_fc").get(4096, "bias")?); candle_nn::Linear::new(w, b) };
        let h_9_mlp_c_proj = { let w = vb.pp("h.9.mlp.c_proj").get((4096, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("h.9.mlp.c_proj").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let h_9_mlp_dropout = Dropout::new();
        let ln_f = candle_nn::layer_norm(1024, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("ln_f"))?;
        let wpe = candle_nn::embedding(config.context_length, config.hidden_dim, vb.pp("wpe"))?;
        let wte = candle_nn::embedding(config.vocab_size, config.hidden_dim, vb.pp("wte"))?;
        Ok(Self { drop, h_0_attn_c_attn, h_0_attn_c_proj, h_0_attn_resid_dropout, h_0_ln_1, h_0_ln_2, h_0_mlp_act, h_0_mlp_c_fc, h_0_mlp_c_proj, h_0_mlp_dropout, h_1_attn_c_attn, h_1_attn_c_proj, h_1_attn_resid_dropout, h_1_ln_1, h_1_ln_2, h_1_mlp_act, h_1_mlp_c_fc, h_1_mlp_c_proj, h_1_mlp_dropout, h_10_attn_c_attn, h_10_attn_c_proj, h_10_attn_resid_dropout, h_10_ln_1, h_10_ln_2, h_10_mlp_act, h_10_mlp_c_fc, h_10_mlp_c_proj, h_10_mlp_dropout, h_11_attn_c_attn, h_11_attn_c_proj, h_11_attn_resid_dropout, h_11_ln_1, h_11_ln_2, h_11_mlp_act, h_11_mlp_c_fc, h_11_mlp_c_proj, h_11_mlp_dropout, h_12_attn_c_attn, h_12_attn_c_proj, h_12_attn_resid_dropout, h_12_ln_1, h_12_ln_2, h_12_mlp_act, h_12_mlp_c_fc, h_12_mlp_c_proj, h_12_mlp_dropout, h_13_attn_c_attn, h_13_attn_c_proj, h_13_attn_resid_dropout, h_13_ln_1, h_13_ln_2, h_13_mlp_act, h_13_mlp_c_fc, h_13_mlp_c_proj, h_13_mlp_dropout, h_14_attn_c_attn, h_14_attn_c_proj, h_14_attn_resid_dropout, h_14_ln_1, h_14_ln_2, h_14_mlp_act, h_14_mlp_c_fc, h_14_mlp_c_proj, h_14_mlp_dropout, h_15_attn_c_attn, h_15_attn_c_proj, h_15_attn_resid_dropout, h_15_ln_1, h_15_ln_2, h_15_mlp_act, h_15_mlp_c_fc, h_15_mlp_c_proj, h_15_mlp_dropout, h_16_attn_c_attn, h_16_attn_c_proj, h_16_attn_resid_dropout, h_16_ln_1, h_16_ln_2, h_16_mlp_act, h_16_mlp_c_fc, h_16_mlp_c_proj, h_16_mlp_dropout, h_17_attn_c_attn, h_17_attn_c_proj, h_17_attn_resid_dropout, h_17_ln_1, h_17_ln_2, h_17_mlp_act, h_17_mlp_c_fc, h_17_mlp_c_proj, h_17_mlp_dropout, h_18_attn_c_attn, h_18_attn_c_proj, h_18_attn_resid_dropout, h_18_ln_1, h_18_ln_2, h_18_mlp_act, h_18_mlp_c_fc, h_18_mlp_c_proj, h_18_mlp_dropout, h_19_attn_c_attn, h_19_attn_c_proj, h_19_attn_resid_dropout, h_19_ln_1, h_19_ln_2, h_19_mlp_act, h_19_mlp_c_fc, h_19_mlp_c_proj, h_19_mlp_dropout, h_2_attn_c_attn, h_2_attn_c_proj, h_2_attn_resid_dropout, h_2_ln_1, h_2_ln_2, h_2_mlp_act, h_2_mlp_c_fc, h_2_mlp_c_proj, h_2_mlp_dropout, h_20_attn_c_attn, h_20_attn_c_proj, h_20_attn_resid_dropout, h_20_ln_1, h_20_ln_2, h_20_mlp_act, h_20_mlp_c_fc, h_20_mlp_c_proj, h_20_mlp_dropout, h_21_attn_c_attn, h_21_attn_c_proj, h_21_attn_resid_dropout, h_21_ln_1, h_21_ln_2, h_21_mlp_act, h_21_mlp_c_fc, h_21_mlp_c_proj, h_21_mlp_dropout, h_22_attn_c_attn, h_22_attn_c_proj, h_22_attn_resid_dropout, h_22_ln_1, h_22_ln_2, h_22_mlp_act, h_22_mlp_c_fc, h_22_mlp_c_proj, h_22_mlp_dropout, h_23_attn_c_attn, h_23_attn_c_proj, h_23_attn_resid_dropout, h_23_ln_1, h_23_ln_2, h_23_mlp_act, h_23_mlp_c_fc, h_23_mlp_c_proj, h_23_mlp_dropout, h_3_attn_c_attn, h_3_attn_c_proj, h_3_attn_resid_dropout, h_3_ln_1, h_3_ln_2, h_3_mlp_act, h_3_mlp_c_fc, h_3_mlp_c_proj, h_3_mlp_dropout, h_4_attn_c_attn, h_4_attn_c_proj, h_4_attn_resid_dropout, h_4_ln_1, h_4_ln_2, h_4_mlp_act, h_4_mlp_c_fc, h_4_mlp_c_proj, h_4_mlp_dropout, h_5_attn_c_attn, h_5_attn_c_proj, h_5_attn_resid_dropout, h_5_ln_1, h_5_ln_2, h_5_mlp_act, h_5_mlp_c_fc, h_5_mlp_c_proj, h_5_mlp_dropout, h_6_attn_c_attn, h_6_attn_c_proj, h_6_attn_resid_dropout, h_6_ln_1, h_6_ln_2, h_6_mlp_act, h_6_mlp_c_fc, h_6_mlp_c_proj, h_6_mlp_dropout, h_7_attn_c_attn, h_7_attn_c_proj, h_7_attn_resid_dropout, h_7_ln_1, h_7_ln_2, h_7_mlp_act, h_7_mlp_c_fc, h_7_mlp_c_proj, h_7_mlp_dropout, h_8_attn_c_attn, h_8_attn_c_proj, h_8_attn_resid_dropout, h_8_ln_1, h_8_ln_2, h_8_mlp_act, h_8_mlp_c_fc, h_8_mlp_c_proj, h_8_mlp_dropout, h_9_attn_c_attn, h_9_attn_c_proj, h_9_attn_resid_dropout, h_9_ln_1, h_9_ln_2, h_9_mlp_act, h_9_mlp_c_fc, h_9_mlp_c_proj, h_9_mlp_dropout, ln_f, wpe, wte, checker })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let size = xs.dims().to_vec();
        let getitem = size[size.len() - 1].clone();
        let view = pycandle_core::ops::reshape(&xs, &vec![-1isize, getitem as isize])?;
        let size_1 = view.dims().to_vec();
        let getitem_1 = size_1[0].clone();
        let getitem_2 = size[size.len() - 1].clone();
        let add = (getitem_2 + 0usize);
        let getattr_1 = view.device();
        let arange = Tensor::arange(0 as i64, add as i64, view.device())?;
        let unsqueeze = arange.unsqueeze(0)?;
        let wte = self.wte.forward(&view)?;
        py_check!(self.checker, "wte", &wte);
        let wpe = self.wpe.forward(&unsqueeze)?;
        py_check!(self.checker, "wpe", &wpe);
        let add_1 = wte.broadcast_add(&wpe)?;
        let getitem_3 = size[size.len() - 1].clone();
        let add_2 = (getitem_3 + 0usize);
        let size_2 = wte.dims().to_vec();
        let getitem_4 = size_2[0].clone();
        let size_3 = wte.dims().to_vec();
        let getitem_5 = size_3[1].clone();
        let add_3 = (getitem_5 + 0usize);
        let sub = (add_2 - getitem_3);
        let gt = (getitem_3 > 1);
        let getattr_2 = wte.dtype();
        let finfo = ();
        let getattr_3 = f32::MIN;
        let getattr_4 = wte.device();
        let full = Tensor::full(getattr_3, vec![getitem_3, getitem_3], unsqueeze.device())?.to_dtype(unsqueeze.dtype())?;
        let size_4 = pycandle_core::ops::dim(&full, -1 as isize)?;
        let arange_1 = Tensor::arange(0i64, size_4 as i64, arange.device())?;
        let add_4 = arange_1.affine(1.0, 0.0)?;
        let size_5 = pycandle_core::ops::dim(&full, -1 as isize)?;
        let view_1 = pycandle_core::ops::reshape(&add_4, &vec![size_5 as isize, 1isize])?;
        let lt = arange_1.lt(&view_1)?;
        let masked_fill_ = pycandle_core::ops::masked_fill(&full, &lt, 0.0)?;
        let to = full.to_dtype(getattr_2)?;
        let gt_1 = (sub > 0);
        let getitem_6 = pycandle_core::ops::index(&to, vec![pycandle_core::ops::IndexItem::None, pycandle_core::ops::IndexItem::None, pycandle_core::ops::IndexItem::Slice(None, None), pycandle_core::ops::IndexItem::Slice(None, None)])?;
        let add_5 = (getitem_3 + sub);
        let expand = getitem_6.broadcast_as((getitem_1, 1, getitem_3, add_5))?;
        let drop = self.drop.forward(&add_1)?;
        py_check!(self.checker, "drop", &drop);
        let getitem_7 = size[1..].to_vec();
        let add_6 = { let mut v: Vec<isize> = vec![-1isize].iter().map(|&x| x as isize).collect(); v.extend(getitem_7.iter().map(|&x| x as isize)); v };
        let size_6 = pycandle_core::ops::dim(&drop, -1 as isize)?;
        let add_7 = { let mut v: Vec<isize> = add_6.iter().map(|&x| x as isize).collect(); v.extend(vec![size_6].iter().map(|&x| x as isize)); v };
        let h_0_ln_1 = self.h_0_ln_1.forward(&drop)?;
        py_check!(self.checker, "h.0.ln_1", &h_0_ln_1);
        let size_7 = h_0_ln_1.dims().to_vec();
        let getitem_8 = size_7[0].clone();
        let getitem_9 = size_7[1].clone();
        let getitem_10 = size_7[2].clone();
        let size_8 = h_0_ln_1.dims().to_vec();
        let getitem_11 = size_8[..size_8.len() - 1].to_vec();
        let add_8 = { let mut v: Vec<isize> = getitem_11.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_0_attn_c_attn_bias = self.h_0_attn_c_attn.bias().unwrap().clone();
        let size_9 = pycandle_core::ops::dim(&h_0_ln_1, -1 as isize)?;
        let view_2 = pycandle_core::ops::reshape(&h_0_ln_1, &vec![-1isize, size_9 as isize])?;
        let h_0_attn_c_attn_weight = self.h_0_attn_c_attn.weight().clone();
        let addmm = h_0_attn_c_attn_bias.broadcast_add(&view_2)?;
        let view_3 = pycandle_core::ops::reshape(&addmm, &add_8.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split = pycandle_core::ops::split(&view_3, 1024 as usize, 0 as usize)?;
        let getitem_12 = split[0].clone();
        let getitem_13 = split[1].clone();
        let getitem_14 = split[2].clone();
        let size_10 = getitem_12.dims().to_vec();
        let getitem_15 = size_10[..size_10.len() - 1].to_vec();
        let add_9 = { let mut v: Vec<isize> = getitem_15.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_4 = pycandle_core::ops::reshape(&getitem_12, &add_9.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute = view_4.permute(0)?;
        let size_11 = getitem_13.dims().to_vec();
        let getitem_16 = size_11[..size_11.len() - 1].to_vec();
        let add_10 = { let mut v: Vec<isize> = getitem_16.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_5 = pycandle_core::ops::reshape(&getitem_13, &add_10.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_1 = view_5.permute(0)?;
        let size_12 = getitem_14.dims().to_vec();
        let getitem_17 = size_12[..size_12.len() - 1].to_vec();
        let add_11 = { let mut v: Vec<isize> = getitem_17.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_6 = pycandle_core::ops::reshape(&getitem_14, &add_11.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_2 = view_6.permute(0)?;
        let scaled_dot_product_attention = pycandle_core::ops::scaled_dot_product_attention(&permute, &permute_1, &permute_2, None, 0.0, false, None)?;
        let transpose = scaled_dot_product_attention.transpose(1, 2)?;
        let contiguous = transpose.contiguous()?;
        let view_7 = pycandle_core::ops::reshape(&contiguous, &vec![getitem_8 as isize, getitem_9 as isize, 1024isize])?;
        let size_13 = view_7.dims().to_vec();
        let getitem_18 = size_13[..size_13.len() - 1].to_vec();
        let add_12 = { let mut v: Vec<isize> = getitem_18.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_0_attn_c_proj_bias = self.h_0_attn_c_proj.bias().unwrap().clone();
        let size_14 = pycandle_core::ops::dim(&view_7, -1 as isize)?;
        let view_8 = pycandle_core::ops::reshape(&view_7, &vec![-1isize, size_14 as isize])?;
        let h_0_attn_c_proj_weight = self.h_0_attn_c_proj.weight().clone();
        let addmm_1 = h_0_attn_c_proj_bias.broadcast_add(&view_8)?;
        let view_9 = pycandle_core::ops::reshape(&addmm_1, &add_12.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_0_attn_resid_dropout = self.h_0_attn_resid_dropout.forward(&view_9)?;
        py_check!(self.checker, "h.0.attn.resid_dropout", &h_0_attn_resid_dropout);
        let add_13 = h_0_attn_resid_dropout.broadcast_add(&drop)?;
        let h_0_ln_2 = self.h_0_ln_2.forward(&add_13)?;
        py_check!(self.checker, "h.0.ln_2", &h_0_ln_2);
        let size_15 = h_0_ln_2.dims().to_vec();
        let getitem_19 = size_15[..size_15.len() - 1].to_vec();
        let add_14 = { let mut v: Vec<isize> = getitem_19.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_0_mlp_c_fc_bias = self.h_0_mlp_c_fc.bias().unwrap().clone();
        let size_16 = pycandle_core::ops::dim(&h_0_ln_2, -1 as isize)?;
        let view_10 = pycandle_core::ops::reshape(&h_0_ln_2, &vec![-1isize, size_16 as isize])?;
        let h_0_mlp_c_fc_weight = self.h_0_mlp_c_fc.weight().clone();
        let addmm_2 = h_0_mlp_c_fc_bias.broadcast_add(&view_10)?;
        let view_11 = pycandle_core::ops::reshape(&addmm_2, &add_14.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul = view_11.affine(0.5f64, 0.0f64)?;
        let pow_1 = view_11.powf(3.0)?;
        let mul_1 = pow_1.affine(0.044715f64, 0.0f64)?;
        let add_15 = view_11.broadcast_add(&mul_1)?;
        let mul_2 = add_15.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh = mul_2.tanh()?;
        let add_16 = tanh.affine(1.0, 0.0)?;
        let mul_3 = mul.broadcast_mul(&add_16)?;
        let size_17 = mul_3.dims().to_vec();
        let getitem_20 = size_17[..size_17.len() - 1].to_vec();
        let add_17 = { let mut v: Vec<isize> = getitem_20.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_0_mlp_c_proj_bias = self.h_0_mlp_c_proj.bias().unwrap().clone();
        let size_18 = pycandle_core::ops::dim(&mul_3, -1 as isize)?;
        let view_12 = pycandle_core::ops::reshape(&mul_3, &vec![-1isize, size_18 as isize])?;
        let h_0_mlp_c_proj_weight = self.h_0_mlp_c_proj.weight().clone();
        let addmm_3 = h_0_mlp_c_proj_bias.broadcast_add(&view_12)?;
        let view_13 = pycandle_core::ops::reshape(&addmm_3, &add_17.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_0_mlp_dropout = self.h_0_mlp_dropout.forward(&view_13)?;
        py_check!(self.checker, "h.0.mlp.dropout", &h_0_mlp_dropout);
        let add_18 = add_13.broadcast_add(&h_0_mlp_dropout)?;
        let h_1_ln_1 = self.h_1_ln_1.forward(&add_18)?;
        py_check!(self.checker, "h.1.ln_1", &h_1_ln_1);
        let size_19 = h_1_ln_1.dims().to_vec();
        let getitem_21 = size_19[0].clone();
        let getitem_22 = size_19[1].clone();
        let getitem_23 = size_19[2].clone();
        let size_20 = h_1_ln_1.dims().to_vec();
        let getitem_24 = size_20[..size_20.len() - 1].to_vec();
        let add_19 = { let mut v: Vec<isize> = getitem_24.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_1_attn_c_attn_bias = self.h_1_attn_c_attn.bias().unwrap().clone();
        let size_21 = pycandle_core::ops::dim(&h_1_ln_1, -1 as isize)?;
        let view_14 = pycandle_core::ops::reshape(&h_1_ln_1, &vec![-1isize, size_21 as isize])?;
        let h_1_attn_c_attn_weight = self.h_1_attn_c_attn.weight().clone();
        let addmm_4 = h_1_attn_c_attn_bias.broadcast_add(&view_14)?;
        let view_15 = pycandle_core::ops::reshape(&addmm_4, &add_19.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_1 = pycandle_core::ops::split(&view_15, 1024 as usize, 0 as usize)?;
        let getitem_25 = split_1[0].clone();
        let getitem_26 = split_1[1].clone();
        let getitem_27 = split_1[2].clone();
        let size_22 = getitem_25.dims().to_vec();
        let getitem_28 = size_22[..size_22.len() - 1].to_vec();
        let add_20 = { let mut v: Vec<isize> = getitem_28.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_16 = pycandle_core::ops::reshape(&getitem_25, &add_20.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_3 = view_16.permute(0)?;
        let size_23 = getitem_26.dims().to_vec();
        let getitem_29 = size_23[..size_23.len() - 1].to_vec();
        let add_21 = { let mut v: Vec<isize> = getitem_29.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_17 = pycandle_core::ops::reshape(&getitem_26, &add_21.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_4 = view_17.permute(0)?;
        let size_24 = getitem_27.dims().to_vec();
        let getitem_30 = size_24[..size_24.len() - 1].to_vec();
        let add_22 = { let mut v: Vec<isize> = getitem_30.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_18 = pycandle_core::ops::reshape(&getitem_27, &add_22.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_5 = view_18.permute(0)?;
        let scaled_dot_product_attention_1 = pycandle_core::ops::scaled_dot_product_attention(&permute_3, &permute_4, &permute_5, None, 0.0, false, None)?;
        let transpose_1 = scaled_dot_product_attention_1.transpose(1, 2)?;
        let contiguous_1 = transpose_1.contiguous()?;
        let view_19 = pycandle_core::ops::reshape(&contiguous_1, &vec![getitem_21 as isize, getitem_22 as isize, 1024isize])?;
        let size_25 = view_19.dims().to_vec();
        let getitem_31 = size_25[..size_25.len() - 1].to_vec();
        let add_23 = { let mut v: Vec<isize> = getitem_31.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_1_attn_c_proj_bias = self.h_1_attn_c_proj.bias().unwrap().clone();
        let size_26 = pycandle_core::ops::dim(&view_19, -1 as isize)?;
        let view_20 = pycandle_core::ops::reshape(&view_19, &vec![-1isize, size_26 as isize])?;
        let h_1_attn_c_proj_weight = self.h_1_attn_c_proj.weight().clone();
        let addmm_5 = h_1_attn_c_proj_bias.broadcast_add(&view_20)?;
        let view_21 = pycandle_core::ops::reshape(&addmm_5, &add_23.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_1_attn_resid_dropout = self.h_1_attn_resid_dropout.forward(&view_21)?;
        py_check!(self.checker, "h.1.attn.resid_dropout", &h_1_attn_resid_dropout);
        let add_24 = h_1_attn_resid_dropout.broadcast_add(&add_18)?;
        let h_1_ln_2 = self.h_1_ln_2.forward(&add_24)?;
        py_check!(self.checker, "h.1.ln_2", &h_1_ln_2);
        let size_27 = h_1_ln_2.dims().to_vec();
        let getitem_32 = size_27[..size_27.len() - 1].to_vec();
        let add_25 = { let mut v: Vec<isize> = getitem_32.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_1_mlp_c_fc_bias = self.h_1_mlp_c_fc.bias().unwrap().clone();
        let size_28 = pycandle_core::ops::dim(&h_1_ln_2, -1 as isize)?;
        let view_22 = pycandle_core::ops::reshape(&h_1_ln_2, &vec![-1isize, size_28 as isize])?;
        let h_1_mlp_c_fc_weight = self.h_1_mlp_c_fc.weight().clone();
        let addmm_6 = h_1_mlp_c_fc_bias.broadcast_add(&view_22)?;
        let view_23 = pycandle_core::ops::reshape(&addmm_6, &add_25.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_4 = view_23.affine(0.5f64, 0.0f64)?;
        let pow_2 = view_23.powf(3.0)?;
        let mul_5 = pow_2.affine(0.044715f64, 0.0f64)?;
        let add_26 = view_23.broadcast_add(&mul_5)?;
        let mul_6 = add_26.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_1 = mul_6.tanh()?;
        let add_27 = tanh_1.affine(1.0, 0.0)?;
        let mul_7 = mul_4.broadcast_mul(&add_27)?;
        let size_29 = mul_7.dims().to_vec();
        let getitem_33 = size_29[..size_29.len() - 1].to_vec();
        let add_28 = { let mut v: Vec<isize> = getitem_33.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_1_mlp_c_proj_bias = self.h_1_mlp_c_proj.bias().unwrap().clone();
        let size_30 = pycandle_core::ops::dim(&mul_7, -1 as isize)?;
        let view_24 = pycandle_core::ops::reshape(&mul_7, &vec![-1isize, size_30 as isize])?;
        let h_1_mlp_c_proj_weight = self.h_1_mlp_c_proj.weight().clone();
        let addmm_7 = h_1_mlp_c_proj_bias.broadcast_add(&view_24)?;
        let view_25 = pycandle_core::ops::reshape(&addmm_7, &add_28.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_1_mlp_dropout = self.h_1_mlp_dropout.forward(&view_25)?;
        py_check!(self.checker, "h.1.mlp.dropout", &h_1_mlp_dropout);
        let add_29 = add_24.broadcast_add(&h_1_mlp_dropout)?;
        let h_2_ln_1 = self.h_2_ln_1.forward(&add_29)?;
        py_check!(self.checker, "h.2.ln_1", &h_2_ln_1);
        let size_31 = h_2_ln_1.dims().to_vec();
        let getitem_34 = size_31[0].clone();
        let getitem_35 = size_31[1].clone();
        let getitem_36 = size_31[2].clone();
        let size_32 = h_2_ln_1.dims().to_vec();
        let getitem_37 = size_32[..size_32.len() - 1].to_vec();
        let add_30 = { let mut v: Vec<isize> = getitem_37.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_2_attn_c_attn_bias = self.h_2_attn_c_attn.bias().unwrap().clone();
        let size_33 = pycandle_core::ops::dim(&h_2_ln_1, -1 as isize)?;
        let view_26 = pycandle_core::ops::reshape(&h_2_ln_1, &vec![-1isize, size_33 as isize])?;
        let h_2_attn_c_attn_weight = self.h_2_attn_c_attn.weight().clone();
        let addmm_8 = h_2_attn_c_attn_bias.broadcast_add(&view_26)?;
        let view_27 = pycandle_core::ops::reshape(&addmm_8, &add_30.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_2 = pycandle_core::ops::split(&view_27, 1024 as usize, 0 as usize)?;
        let getitem_38 = split_2[0].clone();
        let getitem_39 = split_2[1].clone();
        let getitem_40 = split_2[2].clone();
        let size_34 = getitem_38.dims().to_vec();
        let getitem_41 = size_34[..size_34.len() - 1].to_vec();
        let add_31 = { let mut v: Vec<isize> = getitem_41.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_28 = pycandle_core::ops::reshape(&getitem_38, &add_31.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_6 = view_28.permute(0)?;
        let size_35 = getitem_39.dims().to_vec();
        let getitem_42 = size_35[..size_35.len() - 1].to_vec();
        let add_32 = { let mut v: Vec<isize> = getitem_42.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_29 = pycandle_core::ops::reshape(&getitem_39, &add_32.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_7 = view_29.permute(0)?;
        let size_36 = getitem_40.dims().to_vec();
        let getitem_43 = size_36[..size_36.len() - 1].to_vec();
        let add_33 = { let mut v: Vec<isize> = getitem_43.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_30 = pycandle_core::ops::reshape(&getitem_40, &add_33.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_8 = view_30.permute(0)?;
        let scaled_dot_product_attention_2 = pycandle_core::ops::scaled_dot_product_attention(&permute_6, &permute_7, &permute_8, None, 0.0, false, None)?;
        let transpose_2 = scaled_dot_product_attention_2.transpose(1, 2)?;
        let contiguous_2 = transpose_2.contiguous()?;
        let view_31 = pycandle_core::ops::reshape(&contiguous_2, &vec![getitem_34 as isize, getitem_35 as isize, 1024isize])?;
        let size_37 = view_31.dims().to_vec();
        let getitem_44 = size_37[..size_37.len() - 1].to_vec();
        let add_34 = { let mut v: Vec<isize> = getitem_44.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_2_attn_c_proj_bias = self.h_2_attn_c_proj.bias().unwrap().clone();
        let size_38 = pycandle_core::ops::dim(&view_31, -1 as isize)?;
        let view_32 = pycandle_core::ops::reshape(&view_31, &vec![-1isize, size_38 as isize])?;
        let h_2_attn_c_proj_weight = self.h_2_attn_c_proj.weight().clone();
        let addmm_9 = h_2_attn_c_proj_bias.broadcast_add(&view_32)?;
        let view_33 = pycandle_core::ops::reshape(&addmm_9, &add_34.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_2_attn_resid_dropout = self.h_2_attn_resid_dropout.forward(&view_33)?;
        py_check!(self.checker, "h.2.attn.resid_dropout", &h_2_attn_resid_dropout);
        let add_35 = h_2_attn_resid_dropout.broadcast_add(&add_29)?;
        let h_2_ln_2 = self.h_2_ln_2.forward(&add_35)?;
        py_check!(self.checker, "h.2.ln_2", &h_2_ln_2);
        let size_39 = h_2_ln_2.dims().to_vec();
        let getitem_45 = size_39[..size_39.len() - 1].to_vec();
        let add_36 = { let mut v: Vec<isize> = getitem_45.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_2_mlp_c_fc_bias = self.h_2_mlp_c_fc.bias().unwrap().clone();
        let size_40 = pycandle_core::ops::dim(&h_2_ln_2, -1 as isize)?;
        let view_34 = pycandle_core::ops::reshape(&h_2_ln_2, &vec![-1isize, size_40 as isize])?;
        let h_2_mlp_c_fc_weight = self.h_2_mlp_c_fc.weight().clone();
        let addmm_10 = h_2_mlp_c_fc_bias.broadcast_add(&view_34)?;
        let view_35 = pycandle_core::ops::reshape(&addmm_10, &add_36.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_8 = view_35.affine(0.5f64, 0.0f64)?;
        let pow_3 = view_35.powf(3.0)?;
        let mul_9 = pow_3.affine(0.044715f64, 0.0f64)?;
        let add_37 = view_35.broadcast_add(&mul_9)?;
        let mul_10 = add_37.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_2 = mul_10.tanh()?;
        let add_38 = tanh_2.affine(1.0, 0.0)?;
        let mul_11 = mul_8.broadcast_mul(&add_38)?;
        let size_41 = mul_11.dims().to_vec();
        let getitem_46 = size_41[..size_41.len() - 1].to_vec();
        let add_39 = { let mut v: Vec<isize> = getitem_46.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_2_mlp_c_proj_bias = self.h_2_mlp_c_proj.bias().unwrap().clone();
        let size_42 = pycandle_core::ops::dim(&mul_11, -1 as isize)?;
        let view_36 = pycandle_core::ops::reshape(&mul_11, &vec![-1isize, size_42 as isize])?;
        let h_2_mlp_c_proj_weight = self.h_2_mlp_c_proj.weight().clone();
        let addmm_11 = h_2_mlp_c_proj_bias.broadcast_add(&view_36)?;
        let view_37 = pycandle_core::ops::reshape(&addmm_11, &add_39.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_2_mlp_dropout = self.h_2_mlp_dropout.forward(&view_37)?;
        py_check!(self.checker, "h.2.mlp.dropout", &h_2_mlp_dropout);
        let add_40 = add_35.broadcast_add(&h_2_mlp_dropout)?;
        let h_3_ln_1 = self.h_3_ln_1.forward(&add_40)?;
        py_check!(self.checker, "h.3.ln_1", &h_3_ln_1);
        let size_43 = h_3_ln_1.dims().to_vec();
        let getitem_47 = size_43[0].clone();
        let getitem_48 = size_43[1].clone();
        let getitem_49 = size_43[2].clone();
        let size_44 = h_3_ln_1.dims().to_vec();
        let getitem_50 = size_44[..size_44.len() - 1].to_vec();
        let add_41 = { let mut v: Vec<isize> = getitem_50.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_3_attn_c_attn_bias = self.h_3_attn_c_attn.bias().unwrap().clone();
        let size_45 = pycandle_core::ops::dim(&h_3_ln_1, -1 as isize)?;
        let view_38 = pycandle_core::ops::reshape(&h_3_ln_1, &vec![-1isize, size_45 as isize])?;
        let h_3_attn_c_attn_weight = self.h_3_attn_c_attn.weight().clone();
        let addmm_12 = h_3_attn_c_attn_bias.broadcast_add(&view_38)?;
        let view_39 = pycandle_core::ops::reshape(&addmm_12, &add_41.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_3 = pycandle_core::ops::split(&view_39, 1024 as usize, 0 as usize)?;
        let getitem_51 = split_3[0].clone();
        let getitem_52 = split_3[1].clone();
        let getitem_53 = split_3[2].clone();
        let size_46 = getitem_51.dims().to_vec();
        let getitem_54 = size_46[..size_46.len() - 1].to_vec();
        let add_42 = { let mut v: Vec<isize> = getitem_54.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_40 = pycandle_core::ops::reshape(&getitem_51, &add_42.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_9 = view_40.permute(0)?;
        let size_47 = getitem_52.dims().to_vec();
        let getitem_55 = size_47[..size_47.len() - 1].to_vec();
        let add_43 = { let mut v: Vec<isize> = getitem_55.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_41 = pycandle_core::ops::reshape(&getitem_52, &add_43.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_10 = view_41.permute(0)?;
        let size_48 = getitem_53.dims().to_vec();
        let getitem_56 = size_48[..size_48.len() - 1].to_vec();
        let add_44 = { let mut v: Vec<isize> = getitem_56.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_42 = pycandle_core::ops::reshape(&getitem_53, &add_44.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_11 = view_42.permute(0)?;
        let scaled_dot_product_attention_3 = pycandle_core::ops::scaled_dot_product_attention(&permute_9, &permute_10, &permute_11, None, 0.0, false, None)?;
        let transpose_3 = scaled_dot_product_attention_3.transpose(1, 2)?;
        let contiguous_3 = transpose_3.contiguous()?;
        let view_43 = pycandle_core::ops::reshape(&contiguous_3, &vec![getitem_47 as isize, getitem_48 as isize, 1024isize])?;
        let size_49 = view_43.dims().to_vec();
        let getitem_57 = size_49[..size_49.len() - 1].to_vec();
        let add_45 = { let mut v: Vec<isize> = getitem_57.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_3_attn_c_proj_bias = self.h_3_attn_c_proj.bias().unwrap().clone();
        let size_50 = pycandle_core::ops::dim(&view_43, -1 as isize)?;
        let view_44 = pycandle_core::ops::reshape(&view_43, &vec![-1isize, size_50 as isize])?;
        let h_3_attn_c_proj_weight = self.h_3_attn_c_proj.weight().clone();
        let addmm_13 = h_3_attn_c_proj_bias.broadcast_add(&view_44)?;
        let view_45 = pycandle_core::ops::reshape(&addmm_13, &add_45.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_3_attn_resid_dropout = self.h_3_attn_resid_dropout.forward(&view_45)?;
        py_check!(self.checker, "h.3.attn.resid_dropout", &h_3_attn_resid_dropout);
        let add_46 = h_3_attn_resid_dropout.broadcast_add(&add_40)?;
        let h_3_ln_2 = self.h_3_ln_2.forward(&add_46)?;
        py_check!(self.checker, "h.3.ln_2", &h_3_ln_2);
        let size_51 = h_3_ln_2.dims().to_vec();
        let getitem_58 = size_51[..size_51.len() - 1].to_vec();
        let add_47 = { let mut v: Vec<isize> = getitem_58.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_3_mlp_c_fc_bias = self.h_3_mlp_c_fc.bias().unwrap().clone();
        let size_52 = pycandle_core::ops::dim(&h_3_ln_2, -1 as isize)?;
        let view_46 = pycandle_core::ops::reshape(&h_3_ln_2, &vec![-1isize, size_52 as isize])?;
        let h_3_mlp_c_fc_weight = self.h_3_mlp_c_fc.weight().clone();
        let addmm_14 = h_3_mlp_c_fc_bias.broadcast_add(&view_46)?;
        let view_47 = pycandle_core::ops::reshape(&addmm_14, &add_47.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_12 = view_47.affine(0.5f64, 0.0f64)?;
        let pow_4 = view_47.powf(3.0)?;
        let mul_13 = pow_4.affine(0.044715f64, 0.0f64)?;
        let add_48 = view_47.broadcast_add(&mul_13)?;
        let mul_14 = add_48.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_3 = mul_14.tanh()?;
        let add_49 = tanh_3.affine(1.0, 0.0)?;
        let mul_15 = mul_12.broadcast_mul(&add_49)?;
        let size_53 = mul_15.dims().to_vec();
        let getitem_59 = size_53[..size_53.len() - 1].to_vec();
        let add_50 = { let mut v: Vec<isize> = getitem_59.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_3_mlp_c_proj_bias = self.h_3_mlp_c_proj.bias().unwrap().clone();
        let size_54 = pycandle_core::ops::dim(&mul_15, -1 as isize)?;
        let view_48 = pycandle_core::ops::reshape(&mul_15, &vec![-1isize, size_54 as isize])?;
        let h_3_mlp_c_proj_weight = self.h_3_mlp_c_proj.weight().clone();
        let addmm_15 = h_3_mlp_c_proj_bias.broadcast_add(&view_48)?;
        let view_49 = pycandle_core::ops::reshape(&addmm_15, &add_50.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_3_mlp_dropout = self.h_3_mlp_dropout.forward(&view_49)?;
        py_check!(self.checker, "h.3.mlp.dropout", &h_3_mlp_dropout);
        let add_51 = add_46.broadcast_add(&h_3_mlp_dropout)?;
        let h_4_ln_1 = self.h_4_ln_1.forward(&add_51)?;
        py_check!(self.checker, "h.4.ln_1", &h_4_ln_1);
        let size_55 = h_4_ln_1.dims().to_vec();
        let getitem_60 = size_55[0].clone();
        let getitem_61 = size_55[1].clone();
        let getitem_62 = size_55[2].clone();
        let size_56 = h_4_ln_1.dims().to_vec();
        let getitem_63 = size_56[..size_56.len() - 1].to_vec();
        let add_52 = { let mut v: Vec<isize> = getitem_63.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_4_attn_c_attn_bias = self.h_4_attn_c_attn.bias().unwrap().clone();
        let size_57 = pycandle_core::ops::dim(&h_4_ln_1, -1 as isize)?;
        let view_50 = pycandle_core::ops::reshape(&h_4_ln_1, &vec![-1isize, size_57 as isize])?;
        let h_4_attn_c_attn_weight = self.h_4_attn_c_attn.weight().clone();
        let addmm_16 = h_4_attn_c_attn_bias.broadcast_add(&view_50)?;
        let view_51 = pycandle_core::ops::reshape(&addmm_16, &add_52.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_4 = pycandle_core::ops::split(&view_51, 1024 as usize, 0 as usize)?;
        let getitem_64 = split_4[0].clone();
        let getitem_65 = split_4[1].clone();
        let getitem_66 = split_4[2].clone();
        let size_58 = getitem_64.dims().to_vec();
        let getitem_67 = size_58[..size_58.len() - 1].to_vec();
        let add_53 = { let mut v: Vec<isize> = getitem_67.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_52 = pycandle_core::ops::reshape(&getitem_64, &add_53.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_12 = view_52.permute(0)?;
        let size_59 = getitem_65.dims().to_vec();
        let getitem_68 = size_59[..size_59.len() - 1].to_vec();
        let add_54 = { let mut v: Vec<isize> = getitem_68.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_53 = pycandle_core::ops::reshape(&getitem_65, &add_54.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_13 = view_53.permute(0)?;
        let size_60 = getitem_66.dims().to_vec();
        let getitem_69 = size_60[..size_60.len() - 1].to_vec();
        let add_55 = { let mut v: Vec<isize> = getitem_69.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_54 = pycandle_core::ops::reshape(&getitem_66, &add_55.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_14 = view_54.permute(0)?;
        let scaled_dot_product_attention_4 = pycandle_core::ops::scaled_dot_product_attention(&permute_12, &permute_13, &permute_14, None, 0.0, false, None)?;
        let transpose_4 = scaled_dot_product_attention_4.transpose(1, 2)?;
        let contiguous_4 = transpose_4.contiguous()?;
        let view_55 = pycandle_core::ops::reshape(&contiguous_4, &vec![getitem_60 as isize, getitem_61 as isize, 1024isize])?;
        let size_61 = view_55.dims().to_vec();
        let getitem_70 = size_61[..size_61.len() - 1].to_vec();
        let add_56 = { let mut v: Vec<isize> = getitem_70.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_4_attn_c_proj_bias = self.h_4_attn_c_proj.bias().unwrap().clone();
        let size_62 = pycandle_core::ops::dim(&view_55, -1 as isize)?;
        let view_56 = pycandle_core::ops::reshape(&view_55, &vec![-1isize, size_62 as isize])?;
        let h_4_attn_c_proj_weight = self.h_4_attn_c_proj.weight().clone();
        let addmm_17 = h_4_attn_c_proj_bias.broadcast_add(&view_56)?;
        let view_57 = pycandle_core::ops::reshape(&addmm_17, &add_56.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_4_attn_resid_dropout = self.h_4_attn_resid_dropout.forward(&view_57)?;
        py_check!(self.checker, "h.4.attn.resid_dropout", &h_4_attn_resid_dropout);
        let add_57 = h_4_attn_resid_dropout.broadcast_add(&add_51)?;
        let h_4_ln_2 = self.h_4_ln_2.forward(&add_57)?;
        py_check!(self.checker, "h.4.ln_2", &h_4_ln_2);
        let size_63 = h_4_ln_2.dims().to_vec();
        let getitem_71 = size_63[..size_63.len() - 1].to_vec();
        let add_58 = { let mut v: Vec<isize> = getitem_71.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_4_mlp_c_fc_bias = self.h_4_mlp_c_fc.bias().unwrap().clone();
        let size_64 = pycandle_core::ops::dim(&h_4_ln_2, -1 as isize)?;
        let view_58 = pycandle_core::ops::reshape(&h_4_ln_2, &vec![-1isize, size_64 as isize])?;
        let h_4_mlp_c_fc_weight = self.h_4_mlp_c_fc.weight().clone();
        let addmm_18 = h_4_mlp_c_fc_bias.broadcast_add(&view_58)?;
        let view_59 = pycandle_core::ops::reshape(&addmm_18, &add_58.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_16 = view_59.affine(0.5f64, 0.0f64)?;
        let pow_5 = view_59.powf(3.0)?;
        let mul_17 = pow_5.affine(0.044715f64, 0.0f64)?;
        let add_59 = view_59.broadcast_add(&mul_17)?;
        let mul_18 = add_59.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_4 = mul_18.tanh()?;
        let add_60 = tanh_4.affine(1.0, 0.0)?;
        let mul_19 = mul_16.broadcast_mul(&add_60)?;
        let size_65 = mul_19.dims().to_vec();
        let getitem_72 = size_65[..size_65.len() - 1].to_vec();
        let add_61 = { let mut v: Vec<isize> = getitem_72.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_4_mlp_c_proj_bias = self.h_4_mlp_c_proj.bias().unwrap().clone();
        let size_66 = pycandle_core::ops::dim(&mul_19, -1 as isize)?;
        let view_60 = pycandle_core::ops::reshape(&mul_19, &vec![-1isize, size_66 as isize])?;
        let h_4_mlp_c_proj_weight = self.h_4_mlp_c_proj.weight().clone();
        let addmm_19 = h_4_mlp_c_proj_bias.broadcast_add(&view_60)?;
        let view_61 = pycandle_core::ops::reshape(&addmm_19, &add_61.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_4_mlp_dropout = self.h_4_mlp_dropout.forward(&view_61)?;
        py_check!(self.checker, "h.4.mlp.dropout", &h_4_mlp_dropout);
        let add_62 = add_57.broadcast_add(&h_4_mlp_dropout)?;
        let h_5_ln_1 = self.h_5_ln_1.forward(&add_62)?;
        py_check!(self.checker, "h.5.ln_1", &h_5_ln_1);
        let size_67 = h_5_ln_1.dims().to_vec();
        let getitem_73 = size_67[0].clone();
        let getitem_74 = size_67[1].clone();
        let getitem_75 = size_67[2].clone();
        let size_68 = h_5_ln_1.dims().to_vec();
        let getitem_76 = size_68[..size_68.len() - 1].to_vec();
        let add_63 = { let mut v: Vec<isize> = getitem_76.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_5_attn_c_attn_bias = self.h_5_attn_c_attn.bias().unwrap().clone();
        let size_69 = pycandle_core::ops::dim(&h_5_ln_1, -1 as isize)?;
        let view_62 = pycandle_core::ops::reshape(&h_5_ln_1, &vec![-1isize, size_69 as isize])?;
        let h_5_attn_c_attn_weight = self.h_5_attn_c_attn.weight().clone();
        let addmm_20 = h_5_attn_c_attn_bias.broadcast_add(&view_62)?;
        let view_63 = pycandle_core::ops::reshape(&addmm_20, &add_63.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_5 = pycandle_core::ops::split(&view_63, 1024 as usize, 0 as usize)?;
        let getitem_77 = split_5[0].clone();
        let getitem_78 = split_5[1].clone();
        let getitem_79 = split_5[2].clone();
        let size_70 = getitem_77.dims().to_vec();
        let getitem_80 = size_70[..size_70.len() - 1].to_vec();
        let add_64 = { let mut v: Vec<isize> = getitem_80.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_64 = pycandle_core::ops::reshape(&getitem_77, &add_64.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_15 = view_64.permute(0)?;
        let size_71 = getitem_78.dims().to_vec();
        let getitem_81 = size_71[..size_71.len() - 1].to_vec();
        let add_65 = { let mut v: Vec<isize> = getitem_81.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_65 = pycandle_core::ops::reshape(&getitem_78, &add_65.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_16 = view_65.permute(0)?;
        let size_72 = getitem_79.dims().to_vec();
        let getitem_82 = size_72[..size_72.len() - 1].to_vec();
        let add_66 = { let mut v: Vec<isize> = getitem_82.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_66 = pycandle_core::ops::reshape(&getitem_79, &add_66.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_17 = view_66.permute(0)?;
        let scaled_dot_product_attention_5 = pycandle_core::ops::scaled_dot_product_attention(&permute_15, &permute_16, &permute_17, None, 0.0, false, None)?;
        let transpose_5 = scaled_dot_product_attention_5.transpose(1, 2)?;
        let contiguous_5 = transpose_5.contiguous()?;
        let view_67 = pycandle_core::ops::reshape(&contiguous_5, &vec![getitem_73 as isize, getitem_74 as isize, 1024isize])?;
        let size_73 = view_67.dims().to_vec();
        let getitem_83 = size_73[..size_73.len() - 1].to_vec();
        let add_67 = { let mut v: Vec<isize> = getitem_83.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_5_attn_c_proj_bias = self.h_5_attn_c_proj.bias().unwrap().clone();
        let size_74 = pycandle_core::ops::dim(&view_67, -1 as isize)?;
        let view_68 = pycandle_core::ops::reshape(&view_67, &vec![-1isize, size_74 as isize])?;
        let h_5_attn_c_proj_weight = self.h_5_attn_c_proj.weight().clone();
        let addmm_21 = h_5_attn_c_proj_bias.broadcast_add(&view_68)?;
        let view_69 = pycandle_core::ops::reshape(&addmm_21, &add_67.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_5_attn_resid_dropout = self.h_5_attn_resid_dropout.forward(&view_69)?;
        py_check!(self.checker, "h.5.attn.resid_dropout", &h_5_attn_resid_dropout);
        let add_68 = h_5_attn_resid_dropout.broadcast_add(&add_62)?;
        let h_5_ln_2 = self.h_5_ln_2.forward(&add_68)?;
        py_check!(self.checker, "h.5.ln_2", &h_5_ln_2);
        let size_75 = h_5_ln_2.dims().to_vec();
        let getitem_84 = size_75[..size_75.len() - 1].to_vec();
        let add_69 = { let mut v: Vec<isize> = getitem_84.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_5_mlp_c_fc_bias = self.h_5_mlp_c_fc.bias().unwrap().clone();
        let size_76 = pycandle_core::ops::dim(&h_5_ln_2, -1 as isize)?;
        let view_70 = pycandle_core::ops::reshape(&h_5_ln_2, &vec![-1isize, size_76 as isize])?;
        let h_5_mlp_c_fc_weight = self.h_5_mlp_c_fc.weight().clone();
        let addmm_22 = h_5_mlp_c_fc_bias.broadcast_add(&view_70)?;
        let view_71 = pycandle_core::ops::reshape(&addmm_22, &add_69.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_20 = view_71.affine(0.5f64, 0.0f64)?;
        let pow_6 = view_71.powf(3.0)?;
        let mul_21 = pow_6.affine(0.044715f64, 0.0f64)?;
        let add_70 = view_71.broadcast_add(&mul_21)?;
        let mul_22 = add_70.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_5 = mul_22.tanh()?;
        let add_71 = tanh_5.affine(1.0, 0.0)?;
        let mul_23 = mul_20.broadcast_mul(&add_71)?;
        let size_77 = mul_23.dims().to_vec();
        let getitem_85 = size_77[..size_77.len() - 1].to_vec();
        let add_72 = { let mut v: Vec<isize> = getitem_85.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_5_mlp_c_proj_bias = self.h_5_mlp_c_proj.bias().unwrap().clone();
        let size_78 = pycandle_core::ops::dim(&mul_23, -1 as isize)?;
        let view_72 = pycandle_core::ops::reshape(&mul_23, &vec![-1isize, size_78 as isize])?;
        let h_5_mlp_c_proj_weight = self.h_5_mlp_c_proj.weight().clone();
        let addmm_23 = h_5_mlp_c_proj_bias.broadcast_add(&view_72)?;
        let view_73 = pycandle_core::ops::reshape(&addmm_23, &add_72.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_5_mlp_dropout = self.h_5_mlp_dropout.forward(&view_73)?;
        py_check!(self.checker, "h.5.mlp.dropout", &h_5_mlp_dropout);
        let add_73 = add_68.broadcast_add(&h_5_mlp_dropout)?;
        let h_6_ln_1 = self.h_6_ln_1.forward(&add_73)?;
        py_check!(self.checker, "h.6.ln_1", &h_6_ln_1);
        let size_79 = h_6_ln_1.dims().to_vec();
        let getitem_86 = size_79[0].clone();
        let getitem_87 = size_79[1].clone();
        let getitem_88 = size_79[2].clone();
        let size_80 = h_6_ln_1.dims().to_vec();
        let getitem_89 = size_80[..size_80.len() - 1].to_vec();
        let add_74 = { let mut v: Vec<isize> = getitem_89.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_6_attn_c_attn_bias = self.h_6_attn_c_attn.bias().unwrap().clone();
        let size_81 = pycandle_core::ops::dim(&h_6_ln_1, -1 as isize)?;
        let view_74 = pycandle_core::ops::reshape(&h_6_ln_1, &vec![-1isize, size_81 as isize])?;
        let h_6_attn_c_attn_weight = self.h_6_attn_c_attn.weight().clone();
        let addmm_24 = h_6_attn_c_attn_bias.broadcast_add(&view_74)?;
        let view_75 = pycandle_core::ops::reshape(&addmm_24, &add_74.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_6 = pycandle_core::ops::split(&view_75, 1024 as usize, 0 as usize)?;
        let getitem_90 = split_6[0].clone();
        let getitem_91 = split_6[1].clone();
        let getitem_92 = split_6[2].clone();
        let size_82 = getitem_90.dims().to_vec();
        let getitem_93 = size_82[..size_82.len() - 1].to_vec();
        let add_75 = { let mut v: Vec<isize> = getitem_93.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_76 = pycandle_core::ops::reshape(&getitem_90, &add_75.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_18 = view_76.permute(0)?;
        let size_83 = getitem_91.dims().to_vec();
        let getitem_94 = size_83[..size_83.len() - 1].to_vec();
        let add_76 = { let mut v: Vec<isize> = getitem_94.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_77 = pycandle_core::ops::reshape(&getitem_91, &add_76.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_19 = view_77.permute(0)?;
        let size_84 = getitem_92.dims().to_vec();
        let getitem_95 = size_84[..size_84.len() - 1].to_vec();
        let add_77 = { let mut v: Vec<isize> = getitem_95.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_78 = pycandle_core::ops::reshape(&getitem_92, &add_77.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_20 = view_78.permute(0)?;
        let scaled_dot_product_attention_6 = pycandle_core::ops::scaled_dot_product_attention(&permute_18, &permute_19, &permute_20, None, 0.0, false, None)?;
        let transpose_6 = scaled_dot_product_attention_6.transpose(1, 2)?;
        let contiguous_6 = transpose_6.contiguous()?;
        let view_79 = pycandle_core::ops::reshape(&contiguous_6, &vec![getitem_86 as isize, getitem_87 as isize, 1024isize])?;
        let size_85 = view_79.dims().to_vec();
        let getitem_96 = size_85[..size_85.len() - 1].to_vec();
        let add_78 = { let mut v: Vec<isize> = getitem_96.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_6_attn_c_proj_bias = self.h_6_attn_c_proj.bias().unwrap().clone();
        let size_86 = pycandle_core::ops::dim(&view_79, -1 as isize)?;
        let view_80 = pycandle_core::ops::reshape(&view_79, &vec![-1isize, size_86 as isize])?;
        let h_6_attn_c_proj_weight = self.h_6_attn_c_proj.weight().clone();
        let addmm_25 = h_6_attn_c_proj_bias.broadcast_add(&view_80)?;
        let view_81 = pycandle_core::ops::reshape(&addmm_25, &add_78.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_6_attn_resid_dropout = self.h_6_attn_resid_dropout.forward(&view_81)?;
        py_check!(self.checker, "h.6.attn.resid_dropout", &h_6_attn_resid_dropout);
        let add_79 = h_6_attn_resid_dropout.broadcast_add(&add_73)?;
        let h_6_ln_2 = self.h_6_ln_2.forward(&add_79)?;
        py_check!(self.checker, "h.6.ln_2", &h_6_ln_2);
        let size_87 = h_6_ln_2.dims().to_vec();
        let getitem_97 = size_87[..size_87.len() - 1].to_vec();
        let add_80 = { let mut v: Vec<isize> = getitem_97.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_6_mlp_c_fc_bias = self.h_6_mlp_c_fc.bias().unwrap().clone();
        let size_88 = pycandle_core::ops::dim(&h_6_ln_2, -1 as isize)?;
        let view_82 = pycandle_core::ops::reshape(&h_6_ln_2, &vec![-1isize, size_88 as isize])?;
        let h_6_mlp_c_fc_weight = self.h_6_mlp_c_fc.weight().clone();
        let addmm_26 = h_6_mlp_c_fc_bias.broadcast_add(&view_82)?;
        let view_83 = pycandle_core::ops::reshape(&addmm_26, &add_80.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_24 = view_83.affine(0.5f64, 0.0f64)?;
        let pow_7 = view_83.powf(3.0)?;
        let mul_25 = pow_7.affine(0.044715f64, 0.0f64)?;
        let add_81 = view_83.broadcast_add(&mul_25)?;
        let mul_26 = add_81.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_6 = mul_26.tanh()?;
        let add_82 = tanh_6.affine(1.0, 0.0)?;
        let mul_27 = mul_24.broadcast_mul(&add_82)?;
        let size_89 = mul_27.dims().to_vec();
        let getitem_98 = size_89[..size_89.len() - 1].to_vec();
        let add_83 = { let mut v: Vec<isize> = getitem_98.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_6_mlp_c_proj_bias = self.h_6_mlp_c_proj.bias().unwrap().clone();
        let size_90 = pycandle_core::ops::dim(&mul_27, -1 as isize)?;
        let view_84 = pycandle_core::ops::reshape(&mul_27, &vec![-1isize, size_90 as isize])?;
        let h_6_mlp_c_proj_weight = self.h_6_mlp_c_proj.weight().clone();
        let addmm_27 = h_6_mlp_c_proj_bias.broadcast_add(&view_84)?;
        let view_85 = pycandle_core::ops::reshape(&addmm_27, &add_83.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_6_mlp_dropout = self.h_6_mlp_dropout.forward(&view_85)?;
        py_check!(self.checker, "h.6.mlp.dropout", &h_6_mlp_dropout);
        let add_84 = add_79.broadcast_add(&h_6_mlp_dropout)?;
        let h_7_ln_1 = self.h_7_ln_1.forward(&add_84)?;
        py_check!(self.checker, "h.7.ln_1", &h_7_ln_1);
        let size_91 = h_7_ln_1.dims().to_vec();
        let getitem_99 = size_91[0].clone();
        let getitem_100 = size_91[1].clone();
        let getitem_101 = size_91[2].clone();
        let size_92 = h_7_ln_1.dims().to_vec();
        let getitem_102 = size_92[..size_92.len() - 1].to_vec();
        let add_85 = { let mut v: Vec<isize> = getitem_102.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_7_attn_c_attn_bias = self.h_7_attn_c_attn.bias().unwrap().clone();
        let size_93 = pycandle_core::ops::dim(&h_7_ln_1, -1 as isize)?;
        let view_86 = pycandle_core::ops::reshape(&h_7_ln_1, &vec![-1isize, size_93 as isize])?;
        let h_7_attn_c_attn_weight = self.h_7_attn_c_attn.weight().clone();
        let addmm_28 = h_7_attn_c_attn_bias.broadcast_add(&view_86)?;
        let view_87 = pycandle_core::ops::reshape(&addmm_28, &add_85.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_7 = pycandle_core::ops::split(&view_87, 1024 as usize, 0 as usize)?;
        let getitem_103 = split_7[0].clone();
        let getitem_104 = split_7[1].clone();
        let getitem_105 = split_7[2].clone();
        let size_94 = getitem_103.dims().to_vec();
        let getitem_106 = size_94[..size_94.len() - 1].to_vec();
        let add_86 = { let mut v: Vec<isize> = getitem_106.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_88 = pycandle_core::ops::reshape(&getitem_103, &add_86.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_21 = view_88.permute(0)?;
        let size_95 = getitem_104.dims().to_vec();
        let getitem_107 = size_95[..size_95.len() - 1].to_vec();
        let add_87 = { let mut v: Vec<isize> = getitem_107.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_89 = pycandle_core::ops::reshape(&getitem_104, &add_87.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_22 = view_89.permute(0)?;
        let size_96 = getitem_105.dims().to_vec();
        let getitem_108 = size_96[..size_96.len() - 1].to_vec();
        let add_88 = { let mut v: Vec<isize> = getitem_108.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_90 = pycandle_core::ops::reshape(&getitem_105, &add_88.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_23 = view_90.permute(0)?;
        let scaled_dot_product_attention_7 = pycandle_core::ops::scaled_dot_product_attention(&permute_21, &permute_22, &permute_23, None, 0.0, false, None)?;
        let transpose_7 = scaled_dot_product_attention_7.transpose(1, 2)?;
        let contiguous_7 = transpose_7.contiguous()?;
        let view_91 = pycandle_core::ops::reshape(&contiguous_7, &vec![getitem_99 as isize, getitem_100 as isize, 1024isize])?;
        let size_97 = view_91.dims().to_vec();
        let getitem_109 = size_97[..size_97.len() - 1].to_vec();
        let add_89 = { let mut v: Vec<isize> = getitem_109.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_7_attn_c_proj_bias = self.h_7_attn_c_proj.bias().unwrap().clone();
        let size_98 = pycandle_core::ops::dim(&view_91, -1 as isize)?;
        let view_92 = pycandle_core::ops::reshape(&view_91, &vec![-1isize, size_98 as isize])?;
        let h_7_attn_c_proj_weight = self.h_7_attn_c_proj.weight().clone();
        let addmm_29 = h_7_attn_c_proj_bias.broadcast_add(&view_92)?;
        let view_93 = pycandle_core::ops::reshape(&addmm_29, &add_89.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_7_attn_resid_dropout = self.h_7_attn_resid_dropout.forward(&view_93)?;
        py_check!(self.checker, "h.7.attn.resid_dropout", &h_7_attn_resid_dropout);
        let add_90 = h_7_attn_resid_dropout.broadcast_add(&add_84)?;
        let h_7_ln_2 = self.h_7_ln_2.forward(&add_90)?;
        py_check!(self.checker, "h.7.ln_2", &h_7_ln_2);
        let size_99 = h_7_ln_2.dims().to_vec();
        let getitem_110 = size_99[..size_99.len() - 1].to_vec();
        let add_91 = { let mut v: Vec<isize> = getitem_110.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_7_mlp_c_fc_bias = self.h_7_mlp_c_fc.bias().unwrap().clone();
        let size_100 = pycandle_core::ops::dim(&h_7_ln_2, -1 as isize)?;
        let view_94 = pycandle_core::ops::reshape(&h_7_ln_2, &vec![-1isize, size_100 as isize])?;
        let h_7_mlp_c_fc_weight = self.h_7_mlp_c_fc.weight().clone();
        let addmm_30 = h_7_mlp_c_fc_bias.broadcast_add(&view_94)?;
        let view_95 = pycandle_core::ops::reshape(&addmm_30, &add_91.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_28 = view_95.affine(0.5f64, 0.0f64)?;
        let pow_8 = view_95.powf(3.0)?;
        let mul_29 = pow_8.affine(0.044715f64, 0.0f64)?;
        let add_92 = view_95.broadcast_add(&mul_29)?;
        let mul_30 = add_92.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_7 = mul_30.tanh()?;
        let add_93 = tanh_7.affine(1.0, 0.0)?;
        let mul_31 = mul_28.broadcast_mul(&add_93)?;
        let size_101 = mul_31.dims().to_vec();
        let getitem_111 = size_101[..size_101.len() - 1].to_vec();
        let add_94 = { let mut v: Vec<isize> = getitem_111.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_7_mlp_c_proj_bias = self.h_7_mlp_c_proj.bias().unwrap().clone();
        let size_102 = pycandle_core::ops::dim(&mul_31, -1 as isize)?;
        let view_96 = pycandle_core::ops::reshape(&mul_31, &vec![-1isize, size_102 as isize])?;
        let h_7_mlp_c_proj_weight = self.h_7_mlp_c_proj.weight().clone();
        let addmm_31 = h_7_mlp_c_proj_bias.broadcast_add(&view_96)?;
        let view_97 = pycandle_core::ops::reshape(&addmm_31, &add_94.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_7_mlp_dropout = self.h_7_mlp_dropout.forward(&view_97)?;
        py_check!(self.checker, "h.7.mlp.dropout", &h_7_mlp_dropout);
        let add_95 = add_90.broadcast_add(&h_7_mlp_dropout)?;
        let h_8_ln_1 = self.h_8_ln_1.forward(&add_95)?;
        py_check!(self.checker, "h.8.ln_1", &h_8_ln_1);
        let size_103 = h_8_ln_1.dims().to_vec();
        let getitem_112 = size_103[0].clone();
        let getitem_113 = size_103[1].clone();
        let getitem_114 = size_103[2].clone();
        let size_104 = h_8_ln_1.dims().to_vec();
        let getitem_115 = size_104[..size_104.len() - 1].to_vec();
        let add_96 = { let mut v: Vec<isize> = getitem_115.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_8_attn_c_attn_bias = self.h_8_attn_c_attn.bias().unwrap().clone();
        let size_105 = pycandle_core::ops::dim(&h_8_ln_1, -1 as isize)?;
        let view_98 = pycandle_core::ops::reshape(&h_8_ln_1, &vec![-1isize, size_105 as isize])?;
        let h_8_attn_c_attn_weight = self.h_8_attn_c_attn.weight().clone();
        let addmm_32 = h_8_attn_c_attn_bias.broadcast_add(&view_98)?;
        let view_99 = pycandle_core::ops::reshape(&addmm_32, &add_96.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_8 = pycandle_core::ops::split(&view_99, 1024 as usize, 0 as usize)?;
        let getitem_116 = split_8[0].clone();
        let getitem_117 = split_8[1].clone();
        let getitem_118 = split_8[2].clone();
        let size_106 = getitem_116.dims().to_vec();
        let getitem_119 = size_106[..size_106.len() - 1].to_vec();
        let add_97 = { let mut v: Vec<isize> = getitem_119.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_100 = pycandle_core::ops::reshape(&getitem_116, &add_97.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_24 = view_100.permute(0)?;
        let size_107 = getitem_117.dims().to_vec();
        let getitem_120 = size_107[..size_107.len() - 1].to_vec();
        let add_98 = { let mut v: Vec<isize> = getitem_120.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_101 = pycandle_core::ops::reshape(&getitem_117, &add_98.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_25 = view_101.permute(0)?;
        let size_108 = getitem_118.dims().to_vec();
        let getitem_121 = size_108[..size_108.len() - 1].to_vec();
        let add_99 = { let mut v: Vec<isize> = getitem_121.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_102 = pycandle_core::ops::reshape(&getitem_118, &add_99.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_26 = view_102.permute(0)?;
        let scaled_dot_product_attention_8 = pycandle_core::ops::scaled_dot_product_attention(&permute_24, &permute_25, &permute_26, None, 0.0, false, None)?;
        let transpose_8 = scaled_dot_product_attention_8.transpose(1, 2)?;
        let contiguous_8 = transpose_8.contiguous()?;
        let view_103 = pycandle_core::ops::reshape(&contiguous_8, &vec![getitem_112 as isize, getitem_113 as isize, 1024isize])?;
        let size_109 = view_103.dims().to_vec();
        let getitem_122 = size_109[..size_109.len() - 1].to_vec();
        let add_100 = { let mut v: Vec<isize> = getitem_122.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_8_attn_c_proj_bias = self.h_8_attn_c_proj.bias().unwrap().clone();
        let size_110 = pycandle_core::ops::dim(&view_103, -1 as isize)?;
        let view_104 = pycandle_core::ops::reshape(&view_103, &vec![-1isize, size_110 as isize])?;
        let h_8_attn_c_proj_weight = self.h_8_attn_c_proj.weight().clone();
        let addmm_33 = h_8_attn_c_proj_bias.broadcast_add(&view_104)?;
        let view_105 = pycandle_core::ops::reshape(&addmm_33, &add_100.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_8_attn_resid_dropout = self.h_8_attn_resid_dropout.forward(&view_105)?;
        py_check!(self.checker, "h.8.attn.resid_dropout", &h_8_attn_resid_dropout);
        let add_101 = h_8_attn_resid_dropout.broadcast_add(&add_95)?;
        let h_8_ln_2 = self.h_8_ln_2.forward(&add_101)?;
        py_check!(self.checker, "h.8.ln_2", &h_8_ln_2);
        let size_111 = h_8_ln_2.dims().to_vec();
        let getitem_123 = size_111[..size_111.len() - 1].to_vec();
        let add_102 = { let mut v: Vec<isize> = getitem_123.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_8_mlp_c_fc_bias = self.h_8_mlp_c_fc.bias().unwrap().clone();
        let size_112 = pycandle_core::ops::dim(&h_8_ln_2, -1 as isize)?;
        let view_106 = pycandle_core::ops::reshape(&h_8_ln_2, &vec![-1isize, size_112 as isize])?;
        let h_8_mlp_c_fc_weight = self.h_8_mlp_c_fc.weight().clone();
        let addmm_34 = h_8_mlp_c_fc_bias.broadcast_add(&view_106)?;
        let view_107 = pycandle_core::ops::reshape(&addmm_34, &add_102.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_32 = view_107.affine(0.5f64, 0.0f64)?;
        let pow_9 = view_107.powf(3.0)?;
        let mul_33 = pow_9.affine(0.044715f64, 0.0f64)?;
        let add_103 = view_107.broadcast_add(&mul_33)?;
        let mul_34 = add_103.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_8 = mul_34.tanh()?;
        let add_104 = tanh_8.affine(1.0, 0.0)?;
        let mul_35 = mul_32.broadcast_mul(&add_104)?;
        let size_113 = mul_35.dims().to_vec();
        let getitem_124 = size_113[..size_113.len() - 1].to_vec();
        let add_105 = { let mut v: Vec<isize> = getitem_124.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_8_mlp_c_proj_bias = self.h_8_mlp_c_proj.bias().unwrap().clone();
        let size_114 = pycandle_core::ops::dim(&mul_35, -1 as isize)?;
        let view_108 = pycandle_core::ops::reshape(&mul_35, &vec![-1isize, size_114 as isize])?;
        let h_8_mlp_c_proj_weight = self.h_8_mlp_c_proj.weight().clone();
        let addmm_35 = h_8_mlp_c_proj_bias.broadcast_add(&view_108)?;
        let view_109 = pycandle_core::ops::reshape(&addmm_35, &add_105.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_8_mlp_dropout = self.h_8_mlp_dropout.forward(&view_109)?;
        py_check!(self.checker, "h.8.mlp.dropout", &h_8_mlp_dropout);
        let add_106 = add_101.broadcast_add(&h_8_mlp_dropout)?;
        let h_9_ln_1 = self.h_9_ln_1.forward(&add_106)?;
        py_check!(self.checker, "h.9.ln_1", &h_9_ln_1);
        let size_115 = h_9_ln_1.dims().to_vec();
        let getitem_125 = size_115[0].clone();
        let getitem_126 = size_115[1].clone();
        let getitem_127 = size_115[2].clone();
        let size_116 = h_9_ln_1.dims().to_vec();
        let getitem_128 = size_116[..size_116.len() - 1].to_vec();
        let add_107 = { let mut v: Vec<isize> = getitem_128.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_9_attn_c_attn_bias = self.h_9_attn_c_attn.bias().unwrap().clone();
        let size_117 = pycandle_core::ops::dim(&h_9_ln_1, -1 as isize)?;
        let view_110 = pycandle_core::ops::reshape(&h_9_ln_1, &vec![-1isize, size_117 as isize])?;
        let h_9_attn_c_attn_weight = self.h_9_attn_c_attn.weight().clone();
        let addmm_36 = h_9_attn_c_attn_bias.broadcast_add(&view_110)?;
        let view_111 = pycandle_core::ops::reshape(&addmm_36, &add_107.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_9 = pycandle_core::ops::split(&view_111, 1024 as usize, 0 as usize)?;
        let getitem_129 = split_9[0].clone();
        let getitem_130 = split_9[1].clone();
        let getitem_131 = split_9[2].clone();
        let size_118 = getitem_129.dims().to_vec();
        let getitem_132 = size_118[..size_118.len() - 1].to_vec();
        let add_108 = { let mut v: Vec<isize> = getitem_132.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_112 = pycandle_core::ops::reshape(&getitem_129, &add_108.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_27 = view_112.permute(0)?;
        let size_119 = getitem_130.dims().to_vec();
        let getitem_133 = size_119[..size_119.len() - 1].to_vec();
        let add_109 = { let mut v: Vec<isize> = getitem_133.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_113 = pycandle_core::ops::reshape(&getitem_130, &add_109.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_28 = view_113.permute(0)?;
        let size_120 = getitem_131.dims().to_vec();
        let getitem_134 = size_120[..size_120.len() - 1].to_vec();
        let add_110 = { let mut v: Vec<isize> = getitem_134.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_114 = pycandle_core::ops::reshape(&getitem_131, &add_110.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_29 = view_114.permute(0)?;
        let scaled_dot_product_attention_9 = pycandle_core::ops::scaled_dot_product_attention(&permute_27, &permute_28, &permute_29, None, 0.0, false, None)?;
        let transpose_9 = scaled_dot_product_attention_9.transpose(1, 2)?;
        let contiguous_9 = transpose_9.contiguous()?;
        let view_115 = pycandle_core::ops::reshape(&contiguous_9, &vec![getitem_125 as isize, getitem_126 as isize, 1024isize])?;
        let size_121 = view_115.dims().to_vec();
        let getitem_135 = size_121[..size_121.len() - 1].to_vec();
        let add_111 = { let mut v: Vec<isize> = getitem_135.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_9_attn_c_proj_bias = self.h_9_attn_c_proj.bias().unwrap().clone();
        let size_122 = pycandle_core::ops::dim(&view_115, -1 as isize)?;
        let view_116 = pycandle_core::ops::reshape(&view_115, &vec![-1isize, size_122 as isize])?;
        let h_9_attn_c_proj_weight = self.h_9_attn_c_proj.weight().clone();
        let addmm_37 = h_9_attn_c_proj_bias.broadcast_add(&view_116)?;
        let view_117 = pycandle_core::ops::reshape(&addmm_37, &add_111.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_9_attn_resid_dropout = self.h_9_attn_resid_dropout.forward(&view_117)?;
        py_check!(self.checker, "h.9.attn.resid_dropout", &h_9_attn_resid_dropout);
        let add_112 = h_9_attn_resid_dropout.broadcast_add(&add_106)?;
        let h_9_ln_2 = self.h_9_ln_2.forward(&add_112)?;
        py_check!(self.checker, "h.9.ln_2", &h_9_ln_2);
        let size_123 = h_9_ln_2.dims().to_vec();
        let getitem_136 = size_123[..size_123.len() - 1].to_vec();
        let add_113 = { let mut v: Vec<isize> = getitem_136.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_9_mlp_c_fc_bias = self.h_9_mlp_c_fc.bias().unwrap().clone();
        let size_124 = pycandle_core::ops::dim(&h_9_ln_2, -1 as isize)?;
        let view_118 = pycandle_core::ops::reshape(&h_9_ln_2, &vec![-1isize, size_124 as isize])?;
        let h_9_mlp_c_fc_weight = self.h_9_mlp_c_fc.weight().clone();
        let addmm_38 = h_9_mlp_c_fc_bias.broadcast_add(&view_118)?;
        let view_119 = pycandle_core::ops::reshape(&addmm_38, &add_113.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_36 = view_119.affine(0.5f64, 0.0f64)?;
        let pow_10 = view_119.powf(3.0)?;
        let mul_37 = pow_10.affine(0.044715f64, 0.0f64)?;
        let add_114 = view_119.broadcast_add(&mul_37)?;
        let mul_38 = add_114.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_9 = mul_38.tanh()?;
        let add_115 = tanh_9.affine(1.0, 0.0)?;
        let mul_39 = mul_36.broadcast_mul(&add_115)?;
        let size_125 = mul_39.dims().to_vec();
        let getitem_137 = size_125[..size_125.len() - 1].to_vec();
        let add_116 = { let mut v: Vec<isize> = getitem_137.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_9_mlp_c_proj_bias = self.h_9_mlp_c_proj.bias().unwrap().clone();
        let size_126 = pycandle_core::ops::dim(&mul_39, -1 as isize)?;
        let view_120 = pycandle_core::ops::reshape(&mul_39, &vec![-1isize, size_126 as isize])?;
        let h_9_mlp_c_proj_weight = self.h_9_mlp_c_proj.weight().clone();
        let addmm_39 = h_9_mlp_c_proj_bias.broadcast_add(&view_120)?;
        let view_121 = pycandle_core::ops::reshape(&addmm_39, &add_116.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_9_mlp_dropout = self.h_9_mlp_dropout.forward(&view_121)?;
        py_check!(self.checker, "h.9.mlp.dropout", &h_9_mlp_dropout);
        let add_117 = add_112.broadcast_add(&h_9_mlp_dropout)?;
        let h_10_ln_1 = self.h_10_ln_1.forward(&add_117)?;
        py_check!(self.checker, "h.10.ln_1", &h_10_ln_1);
        let size_127 = h_10_ln_1.dims().to_vec();
        let getitem_138 = size_127[0].clone();
        let getitem_139 = size_127[1].clone();
        let getitem_140 = size_127[2].clone();
        let size_128 = h_10_ln_1.dims().to_vec();
        let getitem_141 = size_128[..size_128.len() - 1].to_vec();
        let add_118 = { let mut v: Vec<isize> = getitem_141.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_10_attn_c_attn_bias = self.h_10_attn_c_attn.bias().unwrap().clone();
        let size_129 = pycandle_core::ops::dim(&h_10_ln_1, -1 as isize)?;
        let view_122 = pycandle_core::ops::reshape(&h_10_ln_1, &vec![-1isize, size_129 as isize])?;
        let h_10_attn_c_attn_weight = self.h_10_attn_c_attn.weight().clone();
        let addmm_40 = h_10_attn_c_attn_bias.broadcast_add(&view_122)?;
        let view_123 = pycandle_core::ops::reshape(&addmm_40, &add_118.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_10 = pycandle_core::ops::split(&view_123, 1024 as usize, 0 as usize)?;
        let getitem_142 = split_10[0].clone();
        let getitem_143 = split_10[1].clone();
        let getitem_144 = split_10[2].clone();
        let size_130 = getitem_142.dims().to_vec();
        let getitem_145 = size_130[..size_130.len() - 1].to_vec();
        let add_119 = { let mut v: Vec<isize> = getitem_145.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_124 = pycandle_core::ops::reshape(&getitem_142, &add_119.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_30 = view_124.permute(0)?;
        let size_131 = getitem_143.dims().to_vec();
        let getitem_146 = size_131[..size_131.len() - 1].to_vec();
        let add_120 = { let mut v: Vec<isize> = getitem_146.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_125 = pycandle_core::ops::reshape(&getitem_143, &add_120.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_31 = view_125.permute(0)?;
        let size_132 = getitem_144.dims().to_vec();
        let getitem_147 = size_132[..size_132.len() - 1].to_vec();
        let add_121 = { let mut v: Vec<isize> = getitem_147.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_126 = pycandle_core::ops::reshape(&getitem_144, &add_121.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_32 = view_126.permute(0)?;
        let scaled_dot_product_attention_10 = pycandle_core::ops::scaled_dot_product_attention(&permute_30, &permute_31, &permute_32, None, 0.0, false, None)?;
        let transpose_10 = scaled_dot_product_attention_10.transpose(1, 2)?;
        let contiguous_10 = transpose_10.contiguous()?;
        let view_127 = pycandle_core::ops::reshape(&contiguous_10, &vec![getitem_138 as isize, getitem_139 as isize, 1024isize])?;
        let size_133 = view_127.dims().to_vec();
        let getitem_148 = size_133[..size_133.len() - 1].to_vec();
        let add_122 = { let mut v: Vec<isize> = getitem_148.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_10_attn_c_proj_bias = self.h_10_attn_c_proj.bias().unwrap().clone();
        let size_134 = pycandle_core::ops::dim(&view_127, -1 as isize)?;
        let view_128 = pycandle_core::ops::reshape(&view_127, &vec![-1isize, size_134 as isize])?;
        let h_10_attn_c_proj_weight = self.h_10_attn_c_proj.weight().clone();
        let addmm_41 = h_10_attn_c_proj_bias.broadcast_add(&view_128)?;
        let view_129 = pycandle_core::ops::reshape(&addmm_41, &add_122.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_10_attn_resid_dropout = self.h_10_attn_resid_dropout.forward(&view_129)?;
        py_check!(self.checker, "h.10.attn.resid_dropout", &h_10_attn_resid_dropout);
        let add_123 = h_10_attn_resid_dropout.broadcast_add(&add_117)?;
        let h_10_ln_2 = self.h_10_ln_2.forward(&add_123)?;
        py_check!(self.checker, "h.10.ln_2", &h_10_ln_2);
        let size_135 = h_10_ln_2.dims().to_vec();
        let getitem_149 = size_135[..size_135.len() - 1].to_vec();
        let add_124 = { let mut v: Vec<isize> = getitem_149.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_10_mlp_c_fc_bias = self.h_10_mlp_c_fc.bias().unwrap().clone();
        let size_136 = pycandle_core::ops::dim(&h_10_ln_2, -1 as isize)?;
        let view_130 = pycandle_core::ops::reshape(&h_10_ln_2, &vec![-1isize, size_136 as isize])?;
        let h_10_mlp_c_fc_weight = self.h_10_mlp_c_fc.weight().clone();
        let addmm_42 = h_10_mlp_c_fc_bias.broadcast_add(&view_130)?;
        let view_131 = pycandle_core::ops::reshape(&addmm_42, &add_124.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_40 = view_131.affine(0.5f64, 0.0f64)?;
        let pow_11 = view_131.powf(3.0)?;
        let mul_41 = pow_11.affine(0.044715f64, 0.0f64)?;
        let add_125 = view_131.broadcast_add(&mul_41)?;
        let mul_42 = add_125.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_10 = mul_42.tanh()?;
        let add_126 = tanh_10.affine(1.0, 0.0)?;
        let mul_43 = mul_40.broadcast_mul(&add_126)?;
        let size_137 = mul_43.dims().to_vec();
        let getitem_150 = size_137[..size_137.len() - 1].to_vec();
        let add_127 = { let mut v: Vec<isize> = getitem_150.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_10_mlp_c_proj_bias = self.h_10_mlp_c_proj.bias().unwrap().clone();
        let size_138 = pycandle_core::ops::dim(&mul_43, -1 as isize)?;
        let view_132 = pycandle_core::ops::reshape(&mul_43, &vec![-1isize, size_138 as isize])?;
        let h_10_mlp_c_proj_weight = self.h_10_mlp_c_proj.weight().clone();
        let addmm_43 = h_10_mlp_c_proj_bias.broadcast_add(&view_132)?;
        let view_133 = pycandle_core::ops::reshape(&addmm_43, &add_127.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_10_mlp_dropout = self.h_10_mlp_dropout.forward(&view_133)?;
        py_check!(self.checker, "h.10.mlp.dropout", &h_10_mlp_dropout);
        let add_128 = add_123.broadcast_add(&h_10_mlp_dropout)?;
        let h_11_ln_1 = self.h_11_ln_1.forward(&add_128)?;
        py_check!(self.checker, "h.11.ln_1", &h_11_ln_1);
        let size_139 = h_11_ln_1.dims().to_vec();
        let getitem_151 = size_139[0].clone();
        let getitem_152 = size_139[1].clone();
        let getitem_153 = size_139[2].clone();
        let size_140 = h_11_ln_1.dims().to_vec();
        let getitem_154 = size_140[..size_140.len() - 1].to_vec();
        let add_129 = { let mut v: Vec<isize> = getitem_154.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_11_attn_c_attn_bias = self.h_11_attn_c_attn.bias().unwrap().clone();
        let size_141 = pycandle_core::ops::dim(&h_11_ln_1, -1 as isize)?;
        let view_134 = pycandle_core::ops::reshape(&h_11_ln_1, &vec![-1isize, size_141 as isize])?;
        let h_11_attn_c_attn_weight = self.h_11_attn_c_attn.weight().clone();
        let addmm_44 = h_11_attn_c_attn_bias.broadcast_add(&view_134)?;
        let view_135 = pycandle_core::ops::reshape(&addmm_44, &add_129.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_11 = pycandle_core::ops::split(&view_135, 1024 as usize, 0 as usize)?;
        let getitem_155 = split_11[0].clone();
        let getitem_156 = split_11[1].clone();
        let getitem_157 = split_11[2].clone();
        let size_142 = getitem_155.dims().to_vec();
        let getitem_158 = size_142[..size_142.len() - 1].to_vec();
        let add_130 = { let mut v: Vec<isize> = getitem_158.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_136 = pycandle_core::ops::reshape(&getitem_155, &add_130.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_33 = view_136.permute(0)?;
        let size_143 = getitem_156.dims().to_vec();
        let getitem_159 = size_143[..size_143.len() - 1].to_vec();
        let add_131 = { let mut v: Vec<isize> = getitem_159.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_137 = pycandle_core::ops::reshape(&getitem_156, &add_131.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_34 = view_137.permute(0)?;
        let size_144 = getitem_157.dims().to_vec();
        let getitem_160 = size_144[..size_144.len() - 1].to_vec();
        let add_132 = { let mut v: Vec<isize> = getitem_160.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_138 = pycandle_core::ops::reshape(&getitem_157, &add_132.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_35 = view_138.permute(0)?;
        let scaled_dot_product_attention_11 = pycandle_core::ops::scaled_dot_product_attention(&permute_33, &permute_34, &permute_35, None, 0.0, false, None)?;
        let transpose_11 = scaled_dot_product_attention_11.transpose(1, 2)?;
        let contiguous_11 = transpose_11.contiguous()?;
        let view_139 = pycandle_core::ops::reshape(&contiguous_11, &vec![getitem_151 as isize, getitem_152 as isize, 1024isize])?;
        let size_145 = view_139.dims().to_vec();
        let getitem_161 = size_145[..size_145.len() - 1].to_vec();
        let add_133 = { let mut v: Vec<isize> = getitem_161.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_11_attn_c_proj_bias = self.h_11_attn_c_proj.bias().unwrap().clone();
        let size_146 = pycandle_core::ops::dim(&view_139, -1 as isize)?;
        let view_140 = pycandle_core::ops::reshape(&view_139, &vec![-1isize, size_146 as isize])?;
        let h_11_attn_c_proj_weight = self.h_11_attn_c_proj.weight().clone();
        let addmm_45 = h_11_attn_c_proj_bias.broadcast_add(&view_140)?;
        let view_141 = pycandle_core::ops::reshape(&addmm_45, &add_133.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_11_attn_resid_dropout = self.h_11_attn_resid_dropout.forward(&view_141)?;
        py_check!(self.checker, "h.11.attn.resid_dropout", &h_11_attn_resid_dropout);
        let add_134 = h_11_attn_resid_dropout.broadcast_add(&add_128)?;
        let h_11_ln_2 = self.h_11_ln_2.forward(&add_134)?;
        py_check!(self.checker, "h.11.ln_2", &h_11_ln_2);
        let size_147 = h_11_ln_2.dims().to_vec();
        let getitem_162 = size_147[..size_147.len() - 1].to_vec();
        let add_135 = { let mut v: Vec<isize> = getitem_162.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_11_mlp_c_fc_bias = self.h_11_mlp_c_fc.bias().unwrap().clone();
        let size_148 = pycandle_core::ops::dim(&h_11_ln_2, -1 as isize)?;
        let view_142 = pycandle_core::ops::reshape(&h_11_ln_2, &vec![-1isize, size_148 as isize])?;
        let h_11_mlp_c_fc_weight = self.h_11_mlp_c_fc.weight().clone();
        let addmm_46 = h_11_mlp_c_fc_bias.broadcast_add(&view_142)?;
        let view_143 = pycandle_core::ops::reshape(&addmm_46, &add_135.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_44 = view_143.affine(0.5f64, 0.0f64)?;
        let pow_12 = view_143.powf(3.0)?;
        let mul_45 = pow_12.affine(0.044715f64, 0.0f64)?;
        let add_136 = view_143.broadcast_add(&mul_45)?;
        let mul_46 = add_136.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_11 = mul_46.tanh()?;
        let add_137 = tanh_11.affine(1.0, 0.0)?;
        let mul_47 = mul_44.broadcast_mul(&add_137)?;
        let size_149 = mul_47.dims().to_vec();
        let getitem_163 = size_149[..size_149.len() - 1].to_vec();
        let add_138 = { let mut v: Vec<isize> = getitem_163.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_11_mlp_c_proj_bias = self.h_11_mlp_c_proj.bias().unwrap().clone();
        let size_150 = pycandle_core::ops::dim(&mul_47, -1 as isize)?;
        let view_144 = pycandle_core::ops::reshape(&mul_47, &vec![-1isize, size_150 as isize])?;
        let h_11_mlp_c_proj_weight = self.h_11_mlp_c_proj.weight().clone();
        let addmm_47 = h_11_mlp_c_proj_bias.broadcast_add(&view_144)?;
        let view_145 = pycandle_core::ops::reshape(&addmm_47, &add_138.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_11_mlp_dropout = self.h_11_mlp_dropout.forward(&view_145)?;
        py_check!(self.checker, "h.11.mlp.dropout", &h_11_mlp_dropout);
        let add_139 = add_134.broadcast_add(&h_11_mlp_dropout)?;
        let h_12_ln_1 = self.h_12_ln_1.forward(&add_139)?;
        py_check!(self.checker, "h.12.ln_1", &h_12_ln_1);
        let size_151 = h_12_ln_1.dims().to_vec();
        let getitem_164 = size_151[0].clone();
        let getitem_165 = size_151[1].clone();
        let getitem_166 = size_151[2].clone();
        let size_152 = h_12_ln_1.dims().to_vec();
        let getitem_167 = size_152[..size_152.len() - 1].to_vec();
        let add_140 = { let mut v: Vec<isize> = getitem_167.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_12_attn_c_attn_bias = self.h_12_attn_c_attn.bias().unwrap().clone();
        let size_153 = pycandle_core::ops::dim(&h_12_ln_1, -1 as isize)?;
        let view_146 = pycandle_core::ops::reshape(&h_12_ln_1, &vec![-1isize, size_153 as isize])?;
        let h_12_attn_c_attn_weight = self.h_12_attn_c_attn.weight().clone();
        let addmm_48 = h_12_attn_c_attn_bias.broadcast_add(&view_146)?;
        let view_147 = pycandle_core::ops::reshape(&addmm_48, &add_140.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_12 = pycandle_core::ops::split(&view_147, 1024 as usize, 0 as usize)?;
        let getitem_168 = split_12[0].clone();
        let getitem_169 = split_12[1].clone();
        let getitem_170 = split_12[2].clone();
        let size_154 = getitem_168.dims().to_vec();
        let getitem_171 = size_154[..size_154.len() - 1].to_vec();
        let add_141 = { let mut v: Vec<isize> = getitem_171.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_148 = pycandle_core::ops::reshape(&getitem_168, &add_141.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_36 = view_148.permute(0)?;
        let size_155 = getitem_169.dims().to_vec();
        let getitem_172 = size_155[..size_155.len() - 1].to_vec();
        let add_142 = { let mut v: Vec<isize> = getitem_172.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_149 = pycandle_core::ops::reshape(&getitem_169, &add_142.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_37 = view_149.permute(0)?;
        let size_156 = getitem_170.dims().to_vec();
        let getitem_173 = size_156[..size_156.len() - 1].to_vec();
        let add_143 = { let mut v: Vec<isize> = getitem_173.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_150 = pycandle_core::ops::reshape(&getitem_170, &add_143.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_38 = view_150.permute(0)?;
        let scaled_dot_product_attention_12 = pycandle_core::ops::scaled_dot_product_attention(&permute_36, &permute_37, &permute_38, None, 0.0, false, None)?;
        let transpose_12 = scaled_dot_product_attention_12.transpose(1, 2)?;
        let contiguous_12 = transpose_12.contiguous()?;
        let view_151 = pycandle_core::ops::reshape(&contiguous_12, &vec![getitem_164 as isize, getitem_165 as isize, 1024isize])?;
        let size_157 = view_151.dims().to_vec();
        let getitem_174 = size_157[..size_157.len() - 1].to_vec();
        let add_144 = { let mut v: Vec<isize> = getitem_174.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_12_attn_c_proj_bias = self.h_12_attn_c_proj.bias().unwrap().clone();
        let size_158 = pycandle_core::ops::dim(&view_151, -1 as isize)?;
        let view_152 = pycandle_core::ops::reshape(&view_151, &vec![-1isize, size_158 as isize])?;
        let h_12_attn_c_proj_weight = self.h_12_attn_c_proj.weight().clone();
        let addmm_49 = h_12_attn_c_proj_bias.broadcast_add(&view_152)?;
        let view_153 = pycandle_core::ops::reshape(&addmm_49, &add_144.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_12_attn_resid_dropout = self.h_12_attn_resid_dropout.forward(&view_153)?;
        py_check!(self.checker, "h.12.attn.resid_dropout", &h_12_attn_resid_dropout);
        let add_145 = h_12_attn_resid_dropout.broadcast_add(&add_139)?;
        let h_12_ln_2 = self.h_12_ln_2.forward(&add_145)?;
        py_check!(self.checker, "h.12.ln_2", &h_12_ln_2);
        let size_159 = h_12_ln_2.dims().to_vec();
        let getitem_175 = size_159[..size_159.len() - 1].to_vec();
        let add_146 = { let mut v: Vec<isize> = getitem_175.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_12_mlp_c_fc_bias = self.h_12_mlp_c_fc.bias().unwrap().clone();
        let size_160 = pycandle_core::ops::dim(&h_12_ln_2, -1 as isize)?;
        let view_154 = pycandle_core::ops::reshape(&h_12_ln_2, &vec![-1isize, size_160 as isize])?;
        let h_12_mlp_c_fc_weight = self.h_12_mlp_c_fc.weight().clone();
        let addmm_50 = h_12_mlp_c_fc_bias.broadcast_add(&view_154)?;
        let view_155 = pycandle_core::ops::reshape(&addmm_50, &add_146.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_48 = view_155.affine(0.5f64, 0.0f64)?;
        let pow_13 = view_155.powf(3.0)?;
        let mul_49 = pow_13.affine(0.044715f64, 0.0f64)?;
        let add_147 = view_155.broadcast_add(&mul_49)?;
        let mul_50 = add_147.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_12 = mul_50.tanh()?;
        let add_148 = tanh_12.affine(1.0, 0.0)?;
        let mul_51 = mul_48.broadcast_mul(&add_148)?;
        let size_161 = mul_51.dims().to_vec();
        let getitem_176 = size_161[..size_161.len() - 1].to_vec();
        let add_149 = { let mut v: Vec<isize> = getitem_176.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_12_mlp_c_proj_bias = self.h_12_mlp_c_proj.bias().unwrap().clone();
        let size_162 = pycandle_core::ops::dim(&mul_51, -1 as isize)?;
        let view_156 = pycandle_core::ops::reshape(&mul_51, &vec![-1isize, size_162 as isize])?;
        let h_12_mlp_c_proj_weight = self.h_12_mlp_c_proj.weight().clone();
        let addmm_51 = h_12_mlp_c_proj_bias.broadcast_add(&view_156)?;
        let view_157 = pycandle_core::ops::reshape(&addmm_51, &add_149.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_12_mlp_dropout = self.h_12_mlp_dropout.forward(&view_157)?;
        py_check!(self.checker, "h.12.mlp.dropout", &h_12_mlp_dropout);
        let add_150 = add_145.broadcast_add(&h_12_mlp_dropout)?;
        let h_13_ln_1 = self.h_13_ln_1.forward(&add_150)?;
        py_check!(self.checker, "h.13.ln_1", &h_13_ln_1);
        let size_163 = h_13_ln_1.dims().to_vec();
        let getitem_177 = size_163[0].clone();
        let getitem_178 = size_163[1].clone();
        let getitem_179 = size_163[2].clone();
        let size_164 = h_13_ln_1.dims().to_vec();
        let getitem_180 = size_164[..size_164.len() - 1].to_vec();
        let add_151 = { let mut v: Vec<isize> = getitem_180.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_13_attn_c_attn_bias = self.h_13_attn_c_attn.bias().unwrap().clone();
        let size_165 = pycandle_core::ops::dim(&h_13_ln_1, -1 as isize)?;
        let view_158 = pycandle_core::ops::reshape(&h_13_ln_1, &vec![-1isize, size_165 as isize])?;
        let h_13_attn_c_attn_weight = self.h_13_attn_c_attn.weight().clone();
        let addmm_52 = h_13_attn_c_attn_bias.broadcast_add(&view_158)?;
        let view_159 = pycandle_core::ops::reshape(&addmm_52, &add_151.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_13 = pycandle_core::ops::split(&view_159, 1024 as usize, 0 as usize)?;
        let getitem_181 = split_13[0].clone();
        let getitem_182 = split_13[1].clone();
        let getitem_183 = split_13[2].clone();
        let size_166 = getitem_181.dims().to_vec();
        let getitem_184 = size_166[..size_166.len() - 1].to_vec();
        let add_152 = { let mut v: Vec<isize> = getitem_184.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_160 = pycandle_core::ops::reshape(&getitem_181, &add_152.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_39 = view_160.permute(0)?;
        let size_167 = getitem_182.dims().to_vec();
        let getitem_185 = size_167[..size_167.len() - 1].to_vec();
        let add_153 = { let mut v: Vec<isize> = getitem_185.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_161 = pycandle_core::ops::reshape(&getitem_182, &add_153.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_40 = view_161.permute(0)?;
        let size_168 = getitem_183.dims().to_vec();
        let getitem_186 = size_168[..size_168.len() - 1].to_vec();
        let add_154 = { let mut v: Vec<isize> = getitem_186.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_162 = pycandle_core::ops::reshape(&getitem_183, &add_154.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_41 = view_162.permute(0)?;
        let scaled_dot_product_attention_13 = pycandle_core::ops::scaled_dot_product_attention(&permute_39, &permute_40, &permute_41, None, 0.0, false, None)?;
        let transpose_13 = scaled_dot_product_attention_13.transpose(1, 2)?;
        let contiguous_13 = transpose_13.contiguous()?;
        let view_163 = pycandle_core::ops::reshape(&contiguous_13, &vec![getitem_177 as isize, getitem_178 as isize, 1024isize])?;
        let size_169 = view_163.dims().to_vec();
        let getitem_187 = size_169[..size_169.len() - 1].to_vec();
        let add_155 = { let mut v: Vec<isize> = getitem_187.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_13_attn_c_proj_bias = self.h_13_attn_c_proj.bias().unwrap().clone();
        let size_170 = pycandle_core::ops::dim(&view_163, -1 as isize)?;
        let view_164 = pycandle_core::ops::reshape(&view_163, &vec![-1isize, size_170 as isize])?;
        let h_13_attn_c_proj_weight = self.h_13_attn_c_proj.weight().clone();
        let addmm_53 = h_13_attn_c_proj_bias.broadcast_add(&view_164)?;
        let view_165 = pycandle_core::ops::reshape(&addmm_53, &add_155.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_13_attn_resid_dropout = self.h_13_attn_resid_dropout.forward(&view_165)?;
        py_check!(self.checker, "h.13.attn.resid_dropout", &h_13_attn_resid_dropout);
        let add_156 = h_13_attn_resid_dropout.broadcast_add(&add_150)?;
        let h_13_ln_2 = self.h_13_ln_2.forward(&add_156)?;
        py_check!(self.checker, "h.13.ln_2", &h_13_ln_2);
        let size_171 = h_13_ln_2.dims().to_vec();
        let getitem_188 = size_171[..size_171.len() - 1].to_vec();
        let add_157 = { let mut v: Vec<isize> = getitem_188.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_13_mlp_c_fc_bias = self.h_13_mlp_c_fc.bias().unwrap().clone();
        let size_172 = pycandle_core::ops::dim(&h_13_ln_2, -1 as isize)?;
        let view_166 = pycandle_core::ops::reshape(&h_13_ln_2, &vec![-1isize, size_172 as isize])?;
        let h_13_mlp_c_fc_weight = self.h_13_mlp_c_fc.weight().clone();
        let addmm_54 = h_13_mlp_c_fc_bias.broadcast_add(&view_166)?;
        let view_167 = pycandle_core::ops::reshape(&addmm_54, &add_157.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_52 = view_167.affine(0.5f64, 0.0f64)?;
        let pow_14 = view_167.powf(3.0)?;
        let mul_53 = pow_14.affine(0.044715f64, 0.0f64)?;
        let add_158 = view_167.broadcast_add(&mul_53)?;
        let mul_54 = add_158.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_13 = mul_54.tanh()?;
        let add_159 = tanh_13.affine(1.0, 0.0)?;
        let mul_55 = mul_52.broadcast_mul(&add_159)?;
        let size_173 = mul_55.dims().to_vec();
        let getitem_189 = size_173[..size_173.len() - 1].to_vec();
        let add_160 = { let mut v: Vec<isize> = getitem_189.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_13_mlp_c_proj_bias = self.h_13_mlp_c_proj.bias().unwrap().clone();
        let size_174 = pycandle_core::ops::dim(&mul_55, -1 as isize)?;
        let view_168 = pycandle_core::ops::reshape(&mul_55, &vec![-1isize, size_174 as isize])?;
        let h_13_mlp_c_proj_weight = self.h_13_mlp_c_proj.weight().clone();
        let addmm_55 = h_13_mlp_c_proj_bias.broadcast_add(&view_168)?;
        let view_169 = pycandle_core::ops::reshape(&addmm_55, &add_160.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_13_mlp_dropout = self.h_13_mlp_dropout.forward(&view_169)?;
        py_check!(self.checker, "h.13.mlp.dropout", &h_13_mlp_dropout);
        let add_161 = add_156.broadcast_add(&h_13_mlp_dropout)?;
        let h_14_ln_1 = self.h_14_ln_1.forward(&add_161)?;
        py_check!(self.checker, "h.14.ln_1", &h_14_ln_1);
        let size_175 = h_14_ln_1.dims().to_vec();
        let getitem_190 = size_175[0].clone();
        let getitem_191 = size_175[1].clone();
        let getitem_192 = size_175[2].clone();
        let size_176 = h_14_ln_1.dims().to_vec();
        let getitem_193 = size_176[..size_176.len() - 1].to_vec();
        let add_162 = { let mut v: Vec<isize> = getitem_193.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_14_attn_c_attn_bias = self.h_14_attn_c_attn.bias().unwrap().clone();
        let size_177 = pycandle_core::ops::dim(&h_14_ln_1, -1 as isize)?;
        let view_170 = pycandle_core::ops::reshape(&h_14_ln_1, &vec![-1isize, size_177 as isize])?;
        let h_14_attn_c_attn_weight = self.h_14_attn_c_attn.weight().clone();
        let addmm_56 = h_14_attn_c_attn_bias.broadcast_add(&view_170)?;
        let view_171 = pycandle_core::ops::reshape(&addmm_56, &add_162.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_14 = pycandle_core::ops::split(&view_171, 1024 as usize, 0 as usize)?;
        let getitem_194 = split_14[0].clone();
        let getitem_195 = split_14[1].clone();
        let getitem_196 = split_14[2].clone();
        let size_178 = getitem_194.dims().to_vec();
        let getitem_197 = size_178[..size_178.len() - 1].to_vec();
        let add_163 = { let mut v: Vec<isize> = getitem_197.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_172 = pycandle_core::ops::reshape(&getitem_194, &add_163.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_42 = view_172.permute(0)?;
        let size_179 = getitem_195.dims().to_vec();
        let getitem_198 = size_179[..size_179.len() - 1].to_vec();
        let add_164 = { let mut v: Vec<isize> = getitem_198.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_173 = pycandle_core::ops::reshape(&getitem_195, &add_164.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_43 = view_173.permute(0)?;
        let size_180 = getitem_196.dims().to_vec();
        let getitem_199 = size_180[..size_180.len() - 1].to_vec();
        let add_165 = { let mut v: Vec<isize> = getitem_199.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_174 = pycandle_core::ops::reshape(&getitem_196, &add_165.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_44 = view_174.permute(0)?;
        let scaled_dot_product_attention_14 = pycandle_core::ops::scaled_dot_product_attention(&permute_42, &permute_43, &permute_44, None, 0.0, false, None)?;
        let transpose_14 = scaled_dot_product_attention_14.transpose(1, 2)?;
        let contiguous_14 = transpose_14.contiguous()?;
        let view_175 = pycandle_core::ops::reshape(&contiguous_14, &vec![getitem_190 as isize, getitem_191 as isize, 1024isize])?;
        let size_181 = view_175.dims().to_vec();
        let getitem_200 = size_181[..size_181.len() - 1].to_vec();
        let add_166 = { let mut v: Vec<isize> = getitem_200.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_14_attn_c_proj_bias = self.h_14_attn_c_proj.bias().unwrap().clone();
        let size_182 = pycandle_core::ops::dim(&view_175, -1 as isize)?;
        let view_176 = pycandle_core::ops::reshape(&view_175, &vec![-1isize, size_182 as isize])?;
        let h_14_attn_c_proj_weight = self.h_14_attn_c_proj.weight().clone();
        let addmm_57 = h_14_attn_c_proj_bias.broadcast_add(&view_176)?;
        let view_177 = pycandle_core::ops::reshape(&addmm_57, &add_166.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_14_attn_resid_dropout = self.h_14_attn_resid_dropout.forward(&view_177)?;
        py_check!(self.checker, "h.14.attn.resid_dropout", &h_14_attn_resid_dropout);
        let add_167 = h_14_attn_resid_dropout.broadcast_add(&add_161)?;
        let h_14_ln_2 = self.h_14_ln_2.forward(&add_167)?;
        py_check!(self.checker, "h.14.ln_2", &h_14_ln_2);
        let size_183 = h_14_ln_2.dims().to_vec();
        let getitem_201 = size_183[..size_183.len() - 1].to_vec();
        let add_168 = { let mut v: Vec<isize> = getitem_201.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_14_mlp_c_fc_bias = self.h_14_mlp_c_fc.bias().unwrap().clone();
        let size_184 = pycandle_core::ops::dim(&h_14_ln_2, -1 as isize)?;
        let view_178 = pycandle_core::ops::reshape(&h_14_ln_2, &vec![-1isize, size_184 as isize])?;
        let h_14_mlp_c_fc_weight = self.h_14_mlp_c_fc.weight().clone();
        let addmm_58 = h_14_mlp_c_fc_bias.broadcast_add(&view_178)?;
        let view_179 = pycandle_core::ops::reshape(&addmm_58, &add_168.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_56 = view_179.affine(0.5f64, 0.0f64)?;
        let pow_15 = view_179.powf(3.0)?;
        let mul_57 = pow_15.affine(0.044715f64, 0.0f64)?;
        let add_169 = view_179.broadcast_add(&mul_57)?;
        let mul_58 = add_169.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_14 = mul_58.tanh()?;
        let add_170 = tanh_14.affine(1.0, 0.0)?;
        let mul_59 = mul_56.broadcast_mul(&add_170)?;
        let size_185 = mul_59.dims().to_vec();
        let getitem_202 = size_185[..size_185.len() - 1].to_vec();
        let add_171 = { let mut v: Vec<isize> = getitem_202.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_14_mlp_c_proj_bias = self.h_14_mlp_c_proj.bias().unwrap().clone();
        let size_186 = pycandle_core::ops::dim(&mul_59, -1 as isize)?;
        let view_180 = pycandle_core::ops::reshape(&mul_59, &vec![-1isize, size_186 as isize])?;
        let h_14_mlp_c_proj_weight = self.h_14_mlp_c_proj.weight().clone();
        let addmm_59 = h_14_mlp_c_proj_bias.broadcast_add(&view_180)?;
        let view_181 = pycandle_core::ops::reshape(&addmm_59, &add_171.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_14_mlp_dropout = self.h_14_mlp_dropout.forward(&view_181)?;
        py_check!(self.checker, "h.14.mlp.dropout", &h_14_mlp_dropout);
        let add_172 = add_167.broadcast_add(&h_14_mlp_dropout)?;
        let h_15_ln_1 = self.h_15_ln_1.forward(&add_172)?;
        py_check!(self.checker, "h.15.ln_1", &h_15_ln_1);
        let size_187 = h_15_ln_1.dims().to_vec();
        let getitem_203 = size_187[0].clone();
        let getitem_204 = size_187[1].clone();
        let getitem_205 = size_187[2].clone();
        let size_188 = h_15_ln_1.dims().to_vec();
        let getitem_206 = size_188[..size_188.len() - 1].to_vec();
        let add_173 = { let mut v: Vec<isize> = getitem_206.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_15_attn_c_attn_bias = self.h_15_attn_c_attn.bias().unwrap().clone();
        let size_189 = pycandle_core::ops::dim(&h_15_ln_1, -1 as isize)?;
        let view_182 = pycandle_core::ops::reshape(&h_15_ln_1, &vec![-1isize, size_189 as isize])?;
        let h_15_attn_c_attn_weight = self.h_15_attn_c_attn.weight().clone();
        let addmm_60 = h_15_attn_c_attn_bias.broadcast_add(&view_182)?;
        let view_183 = pycandle_core::ops::reshape(&addmm_60, &add_173.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_15 = pycandle_core::ops::split(&view_183, 1024 as usize, 0 as usize)?;
        let getitem_207 = split_15[0].clone();
        let getitem_208 = split_15[1].clone();
        let getitem_209 = split_15[2].clone();
        let size_190 = getitem_207.dims().to_vec();
        let getitem_210 = size_190[..size_190.len() - 1].to_vec();
        let add_174 = { let mut v: Vec<isize> = getitem_210.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_184 = pycandle_core::ops::reshape(&getitem_207, &add_174.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_45 = view_184.permute(0)?;
        let size_191 = getitem_208.dims().to_vec();
        let getitem_211 = size_191[..size_191.len() - 1].to_vec();
        let add_175 = { let mut v: Vec<isize> = getitem_211.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_185 = pycandle_core::ops::reshape(&getitem_208, &add_175.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_46 = view_185.permute(0)?;
        let size_192 = getitem_209.dims().to_vec();
        let getitem_212 = size_192[..size_192.len() - 1].to_vec();
        let add_176 = { let mut v: Vec<isize> = getitem_212.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_186 = pycandle_core::ops::reshape(&getitem_209, &add_176.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_47 = view_186.permute(0)?;
        let scaled_dot_product_attention_15 = pycandle_core::ops::scaled_dot_product_attention(&permute_45, &permute_46, &permute_47, None, 0.0, false, None)?;
        let transpose_15 = scaled_dot_product_attention_15.transpose(1, 2)?;
        let contiguous_15 = transpose_15.contiguous()?;
        let view_187 = pycandle_core::ops::reshape(&contiguous_15, &vec![getitem_203 as isize, getitem_204 as isize, 1024isize])?;
        let size_193 = view_187.dims().to_vec();
        let getitem_213 = size_193[..size_193.len() - 1].to_vec();
        let add_177 = { let mut v: Vec<isize> = getitem_213.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_15_attn_c_proj_bias = self.h_15_attn_c_proj.bias().unwrap().clone();
        let size_194 = pycandle_core::ops::dim(&view_187, -1 as isize)?;
        let view_188 = pycandle_core::ops::reshape(&view_187, &vec![-1isize, size_194 as isize])?;
        let h_15_attn_c_proj_weight = self.h_15_attn_c_proj.weight().clone();
        let addmm_61 = h_15_attn_c_proj_bias.broadcast_add(&view_188)?;
        let view_189 = pycandle_core::ops::reshape(&addmm_61, &add_177.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_15_attn_resid_dropout = self.h_15_attn_resid_dropout.forward(&view_189)?;
        py_check!(self.checker, "h.15.attn.resid_dropout", &h_15_attn_resid_dropout);
        let add_178 = h_15_attn_resid_dropout.broadcast_add(&add_172)?;
        let h_15_ln_2 = self.h_15_ln_2.forward(&add_178)?;
        py_check!(self.checker, "h.15.ln_2", &h_15_ln_2);
        let size_195 = h_15_ln_2.dims().to_vec();
        let getitem_214 = size_195[..size_195.len() - 1].to_vec();
        let add_179 = { let mut v: Vec<isize> = getitem_214.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_15_mlp_c_fc_bias = self.h_15_mlp_c_fc.bias().unwrap().clone();
        let size_196 = pycandle_core::ops::dim(&h_15_ln_2, -1 as isize)?;
        let view_190 = pycandle_core::ops::reshape(&h_15_ln_2, &vec![-1isize, size_196 as isize])?;
        let h_15_mlp_c_fc_weight = self.h_15_mlp_c_fc.weight().clone();
        let addmm_62 = h_15_mlp_c_fc_bias.broadcast_add(&view_190)?;
        let view_191 = pycandle_core::ops::reshape(&addmm_62, &add_179.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_60 = view_191.affine(0.5f64, 0.0f64)?;
        let pow_16 = view_191.powf(3.0)?;
        let mul_61 = pow_16.affine(0.044715f64, 0.0f64)?;
        let add_180 = view_191.broadcast_add(&mul_61)?;
        let mul_62 = add_180.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_15 = mul_62.tanh()?;
        let add_181 = tanh_15.affine(1.0, 0.0)?;
        let mul_63 = mul_60.broadcast_mul(&add_181)?;
        let size_197 = mul_63.dims().to_vec();
        let getitem_215 = size_197[..size_197.len() - 1].to_vec();
        let add_182 = { let mut v: Vec<isize> = getitem_215.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_15_mlp_c_proj_bias = self.h_15_mlp_c_proj.bias().unwrap().clone();
        let size_198 = pycandle_core::ops::dim(&mul_63, -1 as isize)?;
        let view_192 = pycandle_core::ops::reshape(&mul_63, &vec![-1isize, size_198 as isize])?;
        let h_15_mlp_c_proj_weight = self.h_15_mlp_c_proj.weight().clone();
        let addmm_63 = h_15_mlp_c_proj_bias.broadcast_add(&view_192)?;
        let view_193 = pycandle_core::ops::reshape(&addmm_63, &add_182.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_15_mlp_dropout = self.h_15_mlp_dropout.forward(&view_193)?;
        py_check!(self.checker, "h.15.mlp.dropout", &h_15_mlp_dropout);
        let add_183 = add_178.broadcast_add(&h_15_mlp_dropout)?;
        let h_16_ln_1 = self.h_16_ln_1.forward(&add_183)?;
        py_check!(self.checker, "h.16.ln_1", &h_16_ln_1);
        let size_199 = h_16_ln_1.dims().to_vec();
        let getitem_216 = size_199[0].clone();
        let getitem_217 = size_199[1].clone();
        let getitem_218 = size_199[2].clone();
        let size_200 = h_16_ln_1.dims().to_vec();
        let getitem_219 = size_200[..size_200.len() - 1].to_vec();
        let add_184 = { let mut v: Vec<isize> = getitem_219.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_16_attn_c_attn_bias = self.h_16_attn_c_attn.bias().unwrap().clone();
        let size_201 = pycandle_core::ops::dim(&h_16_ln_1, -1 as isize)?;
        let view_194 = pycandle_core::ops::reshape(&h_16_ln_1, &vec![-1isize, size_201 as isize])?;
        let h_16_attn_c_attn_weight = self.h_16_attn_c_attn.weight().clone();
        let addmm_64 = h_16_attn_c_attn_bias.broadcast_add(&view_194)?;
        let view_195 = pycandle_core::ops::reshape(&addmm_64, &add_184.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_16 = pycandle_core::ops::split(&view_195, 1024 as usize, 0 as usize)?;
        let getitem_220 = split_16[0].clone();
        let getitem_221 = split_16[1].clone();
        let getitem_222 = split_16[2].clone();
        let size_202 = getitem_220.dims().to_vec();
        let getitem_223 = size_202[..size_202.len() - 1].to_vec();
        let add_185 = { let mut v: Vec<isize> = getitem_223.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_196 = pycandle_core::ops::reshape(&getitem_220, &add_185.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_48 = view_196.permute(0)?;
        let size_203 = getitem_221.dims().to_vec();
        let getitem_224 = size_203[..size_203.len() - 1].to_vec();
        let add_186 = { let mut v: Vec<isize> = getitem_224.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_197 = pycandle_core::ops::reshape(&getitem_221, &add_186.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_49 = view_197.permute(0)?;
        let size_204 = getitem_222.dims().to_vec();
        let getitem_225 = size_204[..size_204.len() - 1].to_vec();
        let add_187 = { let mut v: Vec<isize> = getitem_225.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_198 = pycandle_core::ops::reshape(&getitem_222, &add_187.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_50 = view_198.permute(0)?;
        let scaled_dot_product_attention_16 = pycandle_core::ops::scaled_dot_product_attention(&permute_48, &permute_49, &permute_50, None, 0.0, false, None)?;
        let transpose_16 = scaled_dot_product_attention_16.transpose(1, 2)?;
        let contiguous_16 = transpose_16.contiguous()?;
        let view_199 = pycandle_core::ops::reshape(&contiguous_16, &vec![getitem_216 as isize, getitem_217 as isize, 1024isize])?;
        let size_205 = view_199.dims().to_vec();
        let getitem_226 = size_205[..size_205.len() - 1].to_vec();
        let add_188 = { let mut v: Vec<isize> = getitem_226.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_16_attn_c_proj_bias = self.h_16_attn_c_proj.bias().unwrap().clone();
        let size_206 = pycandle_core::ops::dim(&view_199, -1 as isize)?;
        let view_200 = pycandle_core::ops::reshape(&view_199, &vec![-1isize, size_206 as isize])?;
        let h_16_attn_c_proj_weight = self.h_16_attn_c_proj.weight().clone();
        let addmm_65 = h_16_attn_c_proj_bias.broadcast_add(&view_200)?;
        let view_201 = pycandle_core::ops::reshape(&addmm_65, &add_188.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_16_attn_resid_dropout = self.h_16_attn_resid_dropout.forward(&view_201)?;
        py_check!(self.checker, "h.16.attn.resid_dropout", &h_16_attn_resid_dropout);
        let add_189 = h_16_attn_resid_dropout.broadcast_add(&add_183)?;
        let h_16_ln_2 = self.h_16_ln_2.forward(&add_189)?;
        py_check!(self.checker, "h.16.ln_2", &h_16_ln_2);
        let size_207 = h_16_ln_2.dims().to_vec();
        let getitem_227 = size_207[..size_207.len() - 1].to_vec();
        let add_190 = { let mut v: Vec<isize> = getitem_227.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_16_mlp_c_fc_bias = self.h_16_mlp_c_fc.bias().unwrap().clone();
        let size_208 = pycandle_core::ops::dim(&h_16_ln_2, -1 as isize)?;
        let view_202 = pycandle_core::ops::reshape(&h_16_ln_2, &vec![-1isize, size_208 as isize])?;
        let h_16_mlp_c_fc_weight = self.h_16_mlp_c_fc.weight().clone();
        let addmm_66 = h_16_mlp_c_fc_bias.broadcast_add(&view_202)?;
        let view_203 = pycandle_core::ops::reshape(&addmm_66, &add_190.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_64 = view_203.affine(0.5f64, 0.0f64)?;
        let pow_17 = view_203.powf(3.0)?;
        let mul_65 = pow_17.affine(0.044715f64, 0.0f64)?;
        let add_191 = view_203.broadcast_add(&mul_65)?;
        let mul_66 = add_191.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_16 = mul_66.tanh()?;
        let add_192 = tanh_16.affine(1.0, 0.0)?;
        let mul_67 = mul_64.broadcast_mul(&add_192)?;
        let size_209 = mul_67.dims().to_vec();
        let getitem_228 = size_209[..size_209.len() - 1].to_vec();
        let add_193 = { let mut v: Vec<isize> = getitem_228.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_16_mlp_c_proj_bias = self.h_16_mlp_c_proj.bias().unwrap().clone();
        let size_210 = pycandle_core::ops::dim(&mul_67, -1 as isize)?;
        let view_204 = pycandle_core::ops::reshape(&mul_67, &vec![-1isize, size_210 as isize])?;
        let h_16_mlp_c_proj_weight = self.h_16_mlp_c_proj.weight().clone();
        let addmm_67 = h_16_mlp_c_proj_bias.broadcast_add(&view_204)?;
        let view_205 = pycandle_core::ops::reshape(&addmm_67, &add_193.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_16_mlp_dropout = self.h_16_mlp_dropout.forward(&view_205)?;
        py_check!(self.checker, "h.16.mlp.dropout", &h_16_mlp_dropout);
        let add_194 = add_189.broadcast_add(&h_16_mlp_dropout)?;
        let h_17_ln_1 = self.h_17_ln_1.forward(&add_194)?;
        py_check!(self.checker, "h.17.ln_1", &h_17_ln_1);
        let size_211 = h_17_ln_1.dims().to_vec();
        let getitem_229 = size_211[0].clone();
        let getitem_230 = size_211[1].clone();
        let getitem_231 = size_211[2].clone();
        let size_212 = h_17_ln_1.dims().to_vec();
        let getitem_232 = size_212[..size_212.len() - 1].to_vec();
        let add_195 = { let mut v: Vec<isize> = getitem_232.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_17_attn_c_attn_bias = self.h_17_attn_c_attn.bias().unwrap().clone();
        let size_213 = pycandle_core::ops::dim(&h_17_ln_1, -1 as isize)?;
        let view_206 = pycandle_core::ops::reshape(&h_17_ln_1, &vec![-1isize, size_213 as isize])?;
        let h_17_attn_c_attn_weight = self.h_17_attn_c_attn.weight().clone();
        let addmm_68 = h_17_attn_c_attn_bias.broadcast_add(&view_206)?;
        let view_207 = pycandle_core::ops::reshape(&addmm_68, &add_195.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_17 = pycandle_core::ops::split(&view_207, 1024 as usize, 0 as usize)?;
        let getitem_233 = split_17[0].clone();
        let getitem_234 = split_17[1].clone();
        let getitem_235 = split_17[2].clone();
        let size_214 = getitem_233.dims().to_vec();
        let getitem_236 = size_214[..size_214.len() - 1].to_vec();
        let add_196 = { let mut v: Vec<isize> = getitem_236.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_208 = pycandle_core::ops::reshape(&getitem_233, &add_196.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_51 = view_208.permute(0)?;
        let size_215 = getitem_234.dims().to_vec();
        let getitem_237 = size_215[..size_215.len() - 1].to_vec();
        let add_197 = { let mut v: Vec<isize> = getitem_237.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_209 = pycandle_core::ops::reshape(&getitem_234, &add_197.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_52 = view_209.permute(0)?;
        let size_216 = getitem_235.dims().to_vec();
        let getitem_238 = size_216[..size_216.len() - 1].to_vec();
        let add_198 = { let mut v: Vec<isize> = getitem_238.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_210 = pycandle_core::ops::reshape(&getitem_235, &add_198.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_53 = view_210.permute(0)?;
        let scaled_dot_product_attention_17 = pycandle_core::ops::scaled_dot_product_attention(&permute_51, &permute_52, &permute_53, None, 0.0, false, None)?;
        let transpose_17 = scaled_dot_product_attention_17.transpose(1, 2)?;
        let contiguous_17 = transpose_17.contiguous()?;
        let view_211 = pycandle_core::ops::reshape(&contiguous_17, &vec![getitem_229 as isize, getitem_230 as isize, 1024isize])?;
        let size_217 = view_211.dims().to_vec();
        let getitem_239 = size_217[..size_217.len() - 1].to_vec();
        let add_199 = { let mut v: Vec<isize> = getitem_239.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_17_attn_c_proj_bias = self.h_17_attn_c_proj.bias().unwrap().clone();
        let size_218 = pycandle_core::ops::dim(&view_211, -1 as isize)?;
        let view_212 = pycandle_core::ops::reshape(&view_211, &vec![-1isize, size_218 as isize])?;
        let h_17_attn_c_proj_weight = self.h_17_attn_c_proj.weight().clone();
        let addmm_69 = h_17_attn_c_proj_bias.broadcast_add(&view_212)?;
        let view_213 = pycandle_core::ops::reshape(&addmm_69, &add_199.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_17_attn_resid_dropout = self.h_17_attn_resid_dropout.forward(&view_213)?;
        py_check!(self.checker, "h.17.attn.resid_dropout", &h_17_attn_resid_dropout);
        let add_200 = h_17_attn_resid_dropout.broadcast_add(&add_194)?;
        let h_17_ln_2 = self.h_17_ln_2.forward(&add_200)?;
        py_check!(self.checker, "h.17.ln_2", &h_17_ln_2);
        let size_219 = h_17_ln_2.dims().to_vec();
        let getitem_240 = size_219[..size_219.len() - 1].to_vec();
        let add_201 = { let mut v: Vec<isize> = getitem_240.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_17_mlp_c_fc_bias = self.h_17_mlp_c_fc.bias().unwrap().clone();
        let size_220 = pycandle_core::ops::dim(&h_17_ln_2, -1 as isize)?;
        let view_214 = pycandle_core::ops::reshape(&h_17_ln_2, &vec![-1isize, size_220 as isize])?;
        let h_17_mlp_c_fc_weight = self.h_17_mlp_c_fc.weight().clone();
        let addmm_70 = h_17_mlp_c_fc_bias.broadcast_add(&view_214)?;
        let view_215 = pycandle_core::ops::reshape(&addmm_70, &add_201.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_68 = view_215.affine(0.5f64, 0.0f64)?;
        let pow_18 = view_215.powf(3.0)?;
        let mul_69 = pow_18.affine(0.044715f64, 0.0f64)?;
        let add_202 = view_215.broadcast_add(&mul_69)?;
        let mul_70 = add_202.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_17 = mul_70.tanh()?;
        let add_203 = tanh_17.affine(1.0, 0.0)?;
        let mul_71 = mul_68.broadcast_mul(&add_203)?;
        let size_221 = mul_71.dims().to_vec();
        let getitem_241 = size_221[..size_221.len() - 1].to_vec();
        let add_204 = { let mut v: Vec<isize> = getitem_241.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_17_mlp_c_proj_bias = self.h_17_mlp_c_proj.bias().unwrap().clone();
        let size_222 = pycandle_core::ops::dim(&mul_71, -1 as isize)?;
        let view_216 = pycandle_core::ops::reshape(&mul_71, &vec![-1isize, size_222 as isize])?;
        let h_17_mlp_c_proj_weight = self.h_17_mlp_c_proj.weight().clone();
        let addmm_71 = h_17_mlp_c_proj_bias.broadcast_add(&view_216)?;
        let view_217 = pycandle_core::ops::reshape(&addmm_71, &add_204.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_17_mlp_dropout = self.h_17_mlp_dropout.forward(&view_217)?;
        py_check!(self.checker, "h.17.mlp.dropout", &h_17_mlp_dropout);
        let add_205 = add_200.broadcast_add(&h_17_mlp_dropout)?;
        let h_18_ln_1 = self.h_18_ln_1.forward(&add_205)?;
        py_check!(self.checker, "h.18.ln_1", &h_18_ln_1);
        let size_223 = h_18_ln_1.dims().to_vec();
        let getitem_242 = size_223[0].clone();
        let getitem_243 = size_223[1].clone();
        let getitem_244 = size_223[2].clone();
        let size_224 = h_18_ln_1.dims().to_vec();
        let getitem_245 = size_224[..size_224.len() - 1].to_vec();
        let add_206 = { let mut v: Vec<isize> = getitem_245.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_18_attn_c_attn_bias = self.h_18_attn_c_attn.bias().unwrap().clone();
        let size_225 = pycandle_core::ops::dim(&h_18_ln_1, -1 as isize)?;
        let view_218 = pycandle_core::ops::reshape(&h_18_ln_1, &vec![-1isize, size_225 as isize])?;
        let h_18_attn_c_attn_weight = self.h_18_attn_c_attn.weight().clone();
        let addmm_72 = h_18_attn_c_attn_bias.broadcast_add(&view_218)?;
        let view_219 = pycandle_core::ops::reshape(&addmm_72, &add_206.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_18 = pycandle_core::ops::split(&view_219, 1024 as usize, 0 as usize)?;
        let getitem_246 = split_18[0].clone();
        let getitem_247 = split_18[1].clone();
        let getitem_248 = split_18[2].clone();
        let size_226 = getitem_246.dims().to_vec();
        let getitem_249 = size_226[..size_226.len() - 1].to_vec();
        let add_207 = { let mut v: Vec<isize> = getitem_249.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_220 = pycandle_core::ops::reshape(&getitem_246, &add_207.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_54 = view_220.permute(0)?;
        let size_227 = getitem_247.dims().to_vec();
        let getitem_250 = size_227[..size_227.len() - 1].to_vec();
        let add_208 = { let mut v: Vec<isize> = getitem_250.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_221 = pycandle_core::ops::reshape(&getitem_247, &add_208.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_55 = view_221.permute(0)?;
        let size_228 = getitem_248.dims().to_vec();
        let getitem_251 = size_228[..size_228.len() - 1].to_vec();
        let add_209 = { let mut v: Vec<isize> = getitem_251.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_222 = pycandle_core::ops::reshape(&getitem_248, &add_209.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_56 = view_222.permute(0)?;
        let scaled_dot_product_attention_18 = pycandle_core::ops::scaled_dot_product_attention(&permute_54, &permute_55, &permute_56, None, 0.0, false, None)?;
        let transpose_18 = scaled_dot_product_attention_18.transpose(1, 2)?;
        let contiguous_18 = transpose_18.contiguous()?;
        let view_223 = pycandle_core::ops::reshape(&contiguous_18, &vec![getitem_242 as isize, getitem_243 as isize, 1024isize])?;
        let size_229 = view_223.dims().to_vec();
        let getitem_252 = size_229[..size_229.len() - 1].to_vec();
        let add_210 = { let mut v: Vec<isize> = getitem_252.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_18_attn_c_proj_bias = self.h_18_attn_c_proj.bias().unwrap().clone();
        let size_230 = pycandle_core::ops::dim(&view_223, -1 as isize)?;
        let view_224 = pycandle_core::ops::reshape(&view_223, &vec![-1isize, size_230 as isize])?;
        let h_18_attn_c_proj_weight = self.h_18_attn_c_proj.weight().clone();
        let addmm_73 = h_18_attn_c_proj_bias.broadcast_add(&view_224)?;
        let view_225 = pycandle_core::ops::reshape(&addmm_73, &add_210.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_18_attn_resid_dropout = self.h_18_attn_resid_dropout.forward(&view_225)?;
        py_check!(self.checker, "h.18.attn.resid_dropout", &h_18_attn_resid_dropout);
        let add_211 = h_18_attn_resid_dropout.broadcast_add(&add_205)?;
        let h_18_ln_2 = self.h_18_ln_2.forward(&add_211)?;
        py_check!(self.checker, "h.18.ln_2", &h_18_ln_2);
        let size_231 = h_18_ln_2.dims().to_vec();
        let getitem_253 = size_231[..size_231.len() - 1].to_vec();
        let add_212 = { let mut v: Vec<isize> = getitem_253.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_18_mlp_c_fc_bias = self.h_18_mlp_c_fc.bias().unwrap().clone();
        let size_232 = pycandle_core::ops::dim(&h_18_ln_2, -1 as isize)?;
        let view_226 = pycandle_core::ops::reshape(&h_18_ln_2, &vec![-1isize, size_232 as isize])?;
        let h_18_mlp_c_fc_weight = self.h_18_mlp_c_fc.weight().clone();
        let addmm_74 = h_18_mlp_c_fc_bias.broadcast_add(&view_226)?;
        let view_227 = pycandle_core::ops::reshape(&addmm_74, &add_212.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_72 = view_227.affine(0.5f64, 0.0f64)?;
        let pow_19 = view_227.powf(3.0)?;
        let mul_73 = pow_19.affine(0.044715f64, 0.0f64)?;
        let add_213 = view_227.broadcast_add(&mul_73)?;
        let mul_74 = add_213.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_18 = mul_74.tanh()?;
        let add_214 = tanh_18.affine(1.0, 0.0)?;
        let mul_75 = mul_72.broadcast_mul(&add_214)?;
        let size_233 = mul_75.dims().to_vec();
        let getitem_254 = size_233[..size_233.len() - 1].to_vec();
        let add_215 = { let mut v: Vec<isize> = getitem_254.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_18_mlp_c_proj_bias = self.h_18_mlp_c_proj.bias().unwrap().clone();
        let size_234 = pycandle_core::ops::dim(&mul_75, -1 as isize)?;
        let view_228 = pycandle_core::ops::reshape(&mul_75, &vec![-1isize, size_234 as isize])?;
        let h_18_mlp_c_proj_weight = self.h_18_mlp_c_proj.weight().clone();
        let addmm_75 = h_18_mlp_c_proj_bias.broadcast_add(&view_228)?;
        let view_229 = pycandle_core::ops::reshape(&addmm_75, &add_215.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_18_mlp_dropout = self.h_18_mlp_dropout.forward(&view_229)?;
        py_check!(self.checker, "h.18.mlp.dropout", &h_18_mlp_dropout);
        let add_216 = add_211.broadcast_add(&h_18_mlp_dropout)?;
        let h_19_ln_1 = self.h_19_ln_1.forward(&add_216)?;
        py_check!(self.checker, "h.19.ln_1", &h_19_ln_1);
        let size_235 = h_19_ln_1.dims().to_vec();
        let getitem_255 = size_235[0].clone();
        let getitem_256 = size_235[1].clone();
        let getitem_257 = size_235[2].clone();
        let size_236 = h_19_ln_1.dims().to_vec();
        let getitem_258 = size_236[..size_236.len() - 1].to_vec();
        let add_217 = { let mut v: Vec<isize> = getitem_258.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_19_attn_c_attn_bias = self.h_19_attn_c_attn.bias().unwrap().clone();
        let size_237 = pycandle_core::ops::dim(&h_19_ln_1, -1 as isize)?;
        let view_230 = pycandle_core::ops::reshape(&h_19_ln_1, &vec![-1isize, size_237 as isize])?;
        let h_19_attn_c_attn_weight = self.h_19_attn_c_attn.weight().clone();
        let addmm_76 = h_19_attn_c_attn_bias.broadcast_add(&view_230)?;
        let view_231 = pycandle_core::ops::reshape(&addmm_76, &add_217.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_19 = pycandle_core::ops::split(&view_231, 1024 as usize, 0 as usize)?;
        let getitem_259 = split_19[0].clone();
        let getitem_260 = split_19[1].clone();
        let getitem_261 = split_19[2].clone();
        let size_238 = getitem_259.dims().to_vec();
        let getitem_262 = size_238[..size_238.len() - 1].to_vec();
        let add_218 = { let mut v: Vec<isize> = getitem_262.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_232 = pycandle_core::ops::reshape(&getitem_259, &add_218.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_57 = view_232.permute(0)?;
        let size_239 = getitem_260.dims().to_vec();
        let getitem_263 = size_239[..size_239.len() - 1].to_vec();
        let add_219 = { let mut v: Vec<isize> = getitem_263.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_233 = pycandle_core::ops::reshape(&getitem_260, &add_219.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_58 = view_233.permute(0)?;
        let size_240 = getitem_261.dims().to_vec();
        let getitem_264 = size_240[..size_240.len() - 1].to_vec();
        let add_220 = { let mut v: Vec<isize> = getitem_264.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_234 = pycandle_core::ops::reshape(&getitem_261, &add_220.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_59 = view_234.permute(0)?;
        let scaled_dot_product_attention_19 = pycandle_core::ops::scaled_dot_product_attention(&permute_57, &permute_58, &permute_59, None, 0.0, false, None)?;
        let transpose_19 = scaled_dot_product_attention_19.transpose(1, 2)?;
        let contiguous_19 = transpose_19.contiguous()?;
        let view_235 = pycandle_core::ops::reshape(&contiguous_19, &vec![getitem_255 as isize, getitem_256 as isize, 1024isize])?;
        let size_241 = view_235.dims().to_vec();
        let getitem_265 = size_241[..size_241.len() - 1].to_vec();
        let add_221 = { let mut v: Vec<isize> = getitem_265.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_19_attn_c_proj_bias = self.h_19_attn_c_proj.bias().unwrap().clone();
        let size_242 = pycandle_core::ops::dim(&view_235, -1 as isize)?;
        let view_236 = pycandle_core::ops::reshape(&view_235, &vec![-1isize, size_242 as isize])?;
        let h_19_attn_c_proj_weight = self.h_19_attn_c_proj.weight().clone();
        let addmm_77 = h_19_attn_c_proj_bias.broadcast_add(&view_236)?;
        let view_237 = pycandle_core::ops::reshape(&addmm_77, &add_221.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_19_attn_resid_dropout = self.h_19_attn_resid_dropout.forward(&view_237)?;
        py_check!(self.checker, "h.19.attn.resid_dropout", &h_19_attn_resid_dropout);
        let add_222 = h_19_attn_resid_dropout.broadcast_add(&add_216)?;
        let h_19_ln_2 = self.h_19_ln_2.forward(&add_222)?;
        py_check!(self.checker, "h.19.ln_2", &h_19_ln_2);
        let size_243 = h_19_ln_2.dims().to_vec();
        let getitem_266 = size_243[..size_243.len() - 1].to_vec();
        let add_223 = { let mut v: Vec<isize> = getitem_266.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_19_mlp_c_fc_bias = self.h_19_mlp_c_fc.bias().unwrap().clone();
        let size_244 = pycandle_core::ops::dim(&h_19_ln_2, -1 as isize)?;
        let view_238 = pycandle_core::ops::reshape(&h_19_ln_2, &vec![-1isize, size_244 as isize])?;
        let h_19_mlp_c_fc_weight = self.h_19_mlp_c_fc.weight().clone();
        let addmm_78 = h_19_mlp_c_fc_bias.broadcast_add(&view_238)?;
        let view_239 = pycandle_core::ops::reshape(&addmm_78, &add_223.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_76 = view_239.affine(0.5f64, 0.0f64)?;
        let pow_20 = view_239.powf(3.0)?;
        let mul_77 = pow_20.affine(0.044715f64, 0.0f64)?;
        let add_224 = view_239.broadcast_add(&mul_77)?;
        let mul_78 = add_224.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_19 = mul_78.tanh()?;
        let add_225 = tanh_19.affine(1.0, 0.0)?;
        let mul_79 = mul_76.broadcast_mul(&add_225)?;
        let size_245 = mul_79.dims().to_vec();
        let getitem_267 = size_245[..size_245.len() - 1].to_vec();
        let add_226 = { let mut v: Vec<isize> = getitem_267.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_19_mlp_c_proj_bias = self.h_19_mlp_c_proj.bias().unwrap().clone();
        let size_246 = pycandle_core::ops::dim(&mul_79, -1 as isize)?;
        let view_240 = pycandle_core::ops::reshape(&mul_79, &vec![-1isize, size_246 as isize])?;
        let h_19_mlp_c_proj_weight = self.h_19_mlp_c_proj.weight().clone();
        let addmm_79 = h_19_mlp_c_proj_bias.broadcast_add(&view_240)?;
        let view_241 = pycandle_core::ops::reshape(&addmm_79, &add_226.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_19_mlp_dropout = self.h_19_mlp_dropout.forward(&view_241)?;
        py_check!(self.checker, "h.19.mlp.dropout", &h_19_mlp_dropout);
        let add_227 = add_222.broadcast_add(&h_19_mlp_dropout)?;
        let h_20_ln_1 = self.h_20_ln_1.forward(&add_227)?;
        py_check!(self.checker, "h.20.ln_1", &h_20_ln_1);
        let size_247 = h_20_ln_1.dims().to_vec();
        let getitem_268 = size_247[0].clone();
        let getitem_269 = size_247[1].clone();
        let getitem_270 = size_247[2].clone();
        let size_248 = h_20_ln_1.dims().to_vec();
        let getitem_271 = size_248[..size_248.len() - 1].to_vec();
        let add_228 = { let mut v: Vec<isize> = getitem_271.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_20_attn_c_attn_bias = self.h_20_attn_c_attn.bias().unwrap().clone();
        let size_249 = pycandle_core::ops::dim(&h_20_ln_1, -1 as isize)?;
        let view_242 = pycandle_core::ops::reshape(&h_20_ln_1, &vec![-1isize, size_249 as isize])?;
        let h_20_attn_c_attn_weight = self.h_20_attn_c_attn.weight().clone();
        let addmm_80 = h_20_attn_c_attn_bias.broadcast_add(&view_242)?;
        let view_243 = pycandle_core::ops::reshape(&addmm_80, &add_228.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_20 = pycandle_core::ops::split(&view_243, 1024 as usize, 0 as usize)?;
        let getitem_272 = split_20[0].clone();
        let getitem_273 = split_20[1].clone();
        let getitem_274 = split_20[2].clone();
        let size_250 = getitem_272.dims().to_vec();
        let getitem_275 = size_250[..size_250.len() - 1].to_vec();
        let add_229 = { let mut v: Vec<isize> = getitem_275.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_244 = pycandle_core::ops::reshape(&getitem_272, &add_229.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_60 = view_244.permute(0)?;
        let size_251 = getitem_273.dims().to_vec();
        let getitem_276 = size_251[..size_251.len() - 1].to_vec();
        let add_230 = { let mut v: Vec<isize> = getitem_276.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_245 = pycandle_core::ops::reshape(&getitem_273, &add_230.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_61 = view_245.permute(0)?;
        let size_252 = getitem_274.dims().to_vec();
        let getitem_277 = size_252[..size_252.len() - 1].to_vec();
        let add_231 = { let mut v: Vec<isize> = getitem_277.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_246 = pycandle_core::ops::reshape(&getitem_274, &add_231.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_62 = view_246.permute(0)?;
        let scaled_dot_product_attention_20 = pycandle_core::ops::scaled_dot_product_attention(&permute_60, &permute_61, &permute_62, None, 0.0, false, None)?;
        let transpose_20 = scaled_dot_product_attention_20.transpose(1, 2)?;
        let contiguous_20 = transpose_20.contiguous()?;
        let view_247 = pycandle_core::ops::reshape(&contiguous_20, &vec![getitem_268 as isize, getitem_269 as isize, 1024isize])?;
        let size_253 = view_247.dims().to_vec();
        let getitem_278 = size_253[..size_253.len() - 1].to_vec();
        let add_232 = { let mut v: Vec<isize> = getitem_278.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_20_attn_c_proj_bias = self.h_20_attn_c_proj.bias().unwrap().clone();
        let size_254 = pycandle_core::ops::dim(&view_247, -1 as isize)?;
        let view_248 = pycandle_core::ops::reshape(&view_247, &vec![-1isize, size_254 as isize])?;
        let h_20_attn_c_proj_weight = self.h_20_attn_c_proj.weight().clone();
        let addmm_81 = h_20_attn_c_proj_bias.broadcast_add(&view_248)?;
        let view_249 = pycandle_core::ops::reshape(&addmm_81, &add_232.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_20_attn_resid_dropout = self.h_20_attn_resid_dropout.forward(&view_249)?;
        py_check!(self.checker, "h.20.attn.resid_dropout", &h_20_attn_resid_dropout);
        let add_233 = h_20_attn_resid_dropout.broadcast_add(&add_227)?;
        let h_20_ln_2 = self.h_20_ln_2.forward(&add_233)?;
        py_check!(self.checker, "h.20.ln_2", &h_20_ln_2);
        let size_255 = h_20_ln_2.dims().to_vec();
        let getitem_279 = size_255[..size_255.len() - 1].to_vec();
        let add_234 = { let mut v: Vec<isize> = getitem_279.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_20_mlp_c_fc_bias = self.h_20_mlp_c_fc.bias().unwrap().clone();
        let size_256 = pycandle_core::ops::dim(&h_20_ln_2, -1 as isize)?;
        let view_250 = pycandle_core::ops::reshape(&h_20_ln_2, &vec![-1isize, size_256 as isize])?;
        let h_20_mlp_c_fc_weight = self.h_20_mlp_c_fc.weight().clone();
        let addmm_82 = h_20_mlp_c_fc_bias.broadcast_add(&view_250)?;
        let view_251 = pycandle_core::ops::reshape(&addmm_82, &add_234.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_80 = view_251.affine(0.5f64, 0.0f64)?;
        let pow_21 = view_251.powf(3.0)?;
        let mul_81 = pow_21.affine(0.044715f64, 0.0f64)?;
        let add_235 = view_251.broadcast_add(&mul_81)?;
        let mul_82 = add_235.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_20 = mul_82.tanh()?;
        let add_236 = tanh_20.affine(1.0, 0.0)?;
        let mul_83 = mul_80.broadcast_mul(&add_236)?;
        let size_257 = mul_83.dims().to_vec();
        let getitem_280 = size_257[..size_257.len() - 1].to_vec();
        let add_237 = { let mut v: Vec<isize> = getitem_280.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_20_mlp_c_proj_bias = self.h_20_mlp_c_proj.bias().unwrap().clone();
        let size_258 = pycandle_core::ops::dim(&mul_83, -1 as isize)?;
        let view_252 = pycandle_core::ops::reshape(&mul_83, &vec![-1isize, size_258 as isize])?;
        let h_20_mlp_c_proj_weight = self.h_20_mlp_c_proj.weight().clone();
        let addmm_83 = h_20_mlp_c_proj_bias.broadcast_add(&view_252)?;
        let view_253 = pycandle_core::ops::reshape(&addmm_83, &add_237.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_20_mlp_dropout = self.h_20_mlp_dropout.forward(&view_253)?;
        py_check!(self.checker, "h.20.mlp.dropout", &h_20_mlp_dropout);
        let add_238 = add_233.broadcast_add(&h_20_mlp_dropout)?;
        let h_21_ln_1 = self.h_21_ln_1.forward(&add_238)?;
        py_check!(self.checker, "h.21.ln_1", &h_21_ln_1);
        let size_259 = h_21_ln_1.dims().to_vec();
        let getitem_281 = size_259[0].clone();
        let getitem_282 = size_259[1].clone();
        let getitem_283 = size_259[2].clone();
        let size_260 = h_21_ln_1.dims().to_vec();
        let getitem_284 = size_260[..size_260.len() - 1].to_vec();
        let add_239 = { let mut v: Vec<isize> = getitem_284.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_21_attn_c_attn_bias = self.h_21_attn_c_attn.bias().unwrap().clone();
        let size_261 = pycandle_core::ops::dim(&h_21_ln_1, -1 as isize)?;
        let view_254 = pycandle_core::ops::reshape(&h_21_ln_1, &vec![-1isize, size_261 as isize])?;
        let h_21_attn_c_attn_weight = self.h_21_attn_c_attn.weight().clone();
        let addmm_84 = h_21_attn_c_attn_bias.broadcast_add(&view_254)?;
        let view_255 = pycandle_core::ops::reshape(&addmm_84, &add_239.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_21 = pycandle_core::ops::split(&view_255, 1024 as usize, 0 as usize)?;
        let getitem_285 = split_21[0].clone();
        let getitem_286 = split_21[1].clone();
        let getitem_287 = split_21[2].clone();
        let size_262 = getitem_285.dims().to_vec();
        let getitem_288 = size_262[..size_262.len() - 1].to_vec();
        let add_240 = { let mut v: Vec<isize> = getitem_288.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_256 = pycandle_core::ops::reshape(&getitem_285, &add_240.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_63 = view_256.permute(0)?;
        let size_263 = getitem_286.dims().to_vec();
        let getitem_289 = size_263[..size_263.len() - 1].to_vec();
        let add_241 = { let mut v: Vec<isize> = getitem_289.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_257 = pycandle_core::ops::reshape(&getitem_286, &add_241.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_64 = view_257.permute(0)?;
        let size_264 = getitem_287.dims().to_vec();
        let getitem_290 = size_264[..size_264.len() - 1].to_vec();
        let add_242 = { let mut v: Vec<isize> = getitem_290.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_258 = pycandle_core::ops::reshape(&getitem_287, &add_242.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_65 = view_258.permute(0)?;
        let scaled_dot_product_attention_21 = pycandle_core::ops::scaled_dot_product_attention(&permute_63, &permute_64, &permute_65, None, 0.0, false, None)?;
        let transpose_21 = scaled_dot_product_attention_21.transpose(1, 2)?;
        let contiguous_21 = transpose_21.contiguous()?;
        let view_259 = pycandle_core::ops::reshape(&contiguous_21, &vec![getitem_281 as isize, getitem_282 as isize, 1024isize])?;
        let size_265 = view_259.dims().to_vec();
        let getitem_291 = size_265[..size_265.len() - 1].to_vec();
        let add_243 = { let mut v: Vec<isize> = getitem_291.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_21_attn_c_proj_bias = self.h_21_attn_c_proj.bias().unwrap().clone();
        let size_266 = pycandle_core::ops::dim(&view_259, -1 as isize)?;
        let view_260 = pycandle_core::ops::reshape(&view_259, &vec![-1isize, size_266 as isize])?;
        let h_21_attn_c_proj_weight = self.h_21_attn_c_proj.weight().clone();
        let addmm_85 = h_21_attn_c_proj_bias.broadcast_add(&view_260)?;
        let view_261 = pycandle_core::ops::reshape(&addmm_85, &add_243.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_21_attn_resid_dropout = self.h_21_attn_resid_dropout.forward(&view_261)?;
        py_check!(self.checker, "h.21.attn.resid_dropout", &h_21_attn_resid_dropout);
        let add_244 = h_21_attn_resid_dropout.broadcast_add(&add_238)?;
        let h_21_ln_2 = self.h_21_ln_2.forward(&add_244)?;
        py_check!(self.checker, "h.21.ln_2", &h_21_ln_2);
        let size_267 = h_21_ln_2.dims().to_vec();
        let getitem_292 = size_267[..size_267.len() - 1].to_vec();
        let add_245 = { let mut v: Vec<isize> = getitem_292.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_21_mlp_c_fc_bias = self.h_21_mlp_c_fc.bias().unwrap().clone();
        let size_268 = pycandle_core::ops::dim(&h_21_ln_2, -1 as isize)?;
        let view_262 = pycandle_core::ops::reshape(&h_21_ln_2, &vec![-1isize, size_268 as isize])?;
        let h_21_mlp_c_fc_weight = self.h_21_mlp_c_fc.weight().clone();
        let addmm_86 = h_21_mlp_c_fc_bias.broadcast_add(&view_262)?;
        let view_263 = pycandle_core::ops::reshape(&addmm_86, &add_245.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_84 = view_263.affine(0.5f64, 0.0f64)?;
        let pow_22 = view_263.powf(3.0)?;
        let mul_85 = pow_22.affine(0.044715f64, 0.0f64)?;
        let add_246 = view_263.broadcast_add(&mul_85)?;
        let mul_86 = add_246.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_21 = mul_86.tanh()?;
        let add_247 = tanh_21.affine(1.0, 0.0)?;
        let mul_87 = mul_84.broadcast_mul(&add_247)?;
        let size_269 = mul_87.dims().to_vec();
        let getitem_293 = size_269[..size_269.len() - 1].to_vec();
        let add_248 = { let mut v: Vec<isize> = getitem_293.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_21_mlp_c_proj_bias = self.h_21_mlp_c_proj.bias().unwrap().clone();
        let size_270 = pycandle_core::ops::dim(&mul_87, -1 as isize)?;
        let view_264 = pycandle_core::ops::reshape(&mul_87, &vec![-1isize, size_270 as isize])?;
        let h_21_mlp_c_proj_weight = self.h_21_mlp_c_proj.weight().clone();
        let addmm_87 = h_21_mlp_c_proj_bias.broadcast_add(&view_264)?;
        let view_265 = pycandle_core::ops::reshape(&addmm_87, &add_248.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_21_mlp_dropout = self.h_21_mlp_dropout.forward(&view_265)?;
        py_check!(self.checker, "h.21.mlp.dropout", &h_21_mlp_dropout);
        let add_249 = add_244.broadcast_add(&h_21_mlp_dropout)?;
        let h_22_ln_1 = self.h_22_ln_1.forward(&add_249)?;
        py_check!(self.checker, "h.22.ln_1", &h_22_ln_1);
        let size_271 = h_22_ln_1.dims().to_vec();
        let getitem_294 = size_271[0].clone();
        let getitem_295 = size_271[1].clone();
        let getitem_296 = size_271[2].clone();
        let size_272 = h_22_ln_1.dims().to_vec();
        let getitem_297 = size_272[..size_272.len() - 1].to_vec();
        let add_250 = { let mut v: Vec<isize> = getitem_297.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_22_attn_c_attn_bias = self.h_22_attn_c_attn.bias().unwrap().clone();
        let size_273 = pycandle_core::ops::dim(&h_22_ln_1, -1 as isize)?;
        let view_266 = pycandle_core::ops::reshape(&h_22_ln_1, &vec![-1isize, size_273 as isize])?;
        let h_22_attn_c_attn_weight = self.h_22_attn_c_attn.weight().clone();
        let addmm_88 = h_22_attn_c_attn_bias.broadcast_add(&view_266)?;
        let view_267 = pycandle_core::ops::reshape(&addmm_88, &add_250.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_22 = pycandle_core::ops::split(&view_267, 1024 as usize, 0 as usize)?;
        let getitem_298 = split_22[0].clone();
        let getitem_299 = split_22[1].clone();
        let getitem_300 = split_22[2].clone();
        let size_274 = getitem_298.dims().to_vec();
        let getitem_301 = size_274[..size_274.len() - 1].to_vec();
        let add_251 = { let mut v: Vec<isize> = getitem_301.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_268 = pycandle_core::ops::reshape(&getitem_298, &add_251.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_66 = view_268.permute(0)?;
        let size_275 = getitem_299.dims().to_vec();
        let getitem_302 = size_275[..size_275.len() - 1].to_vec();
        let add_252 = { let mut v: Vec<isize> = getitem_302.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_269 = pycandle_core::ops::reshape(&getitem_299, &add_252.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_67 = view_269.permute(0)?;
        let size_276 = getitem_300.dims().to_vec();
        let getitem_303 = size_276[..size_276.len() - 1].to_vec();
        let add_253 = { let mut v: Vec<isize> = getitem_303.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_270 = pycandle_core::ops::reshape(&getitem_300, &add_253.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_68 = view_270.permute(0)?;
        let scaled_dot_product_attention_22 = pycandle_core::ops::scaled_dot_product_attention(&permute_66, &permute_67, &permute_68, None, 0.0, false, None)?;
        let transpose_22 = scaled_dot_product_attention_22.transpose(1, 2)?;
        let contiguous_22 = transpose_22.contiguous()?;
        let view_271 = pycandle_core::ops::reshape(&contiguous_22, &vec![getitem_294 as isize, getitem_295 as isize, 1024isize])?;
        let size_277 = view_271.dims().to_vec();
        let getitem_304 = size_277[..size_277.len() - 1].to_vec();
        let add_254 = { let mut v: Vec<isize> = getitem_304.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_22_attn_c_proj_bias = self.h_22_attn_c_proj.bias().unwrap().clone();
        let size_278 = pycandle_core::ops::dim(&view_271, -1 as isize)?;
        let view_272 = pycandle_core::ops::reshape(&view_271, &vec![-1isize, size_278 as isize])?;
        let h_22_attn_c_proj_weight = self.h_22_attn_c_proj.weight().clone();
        let addmm_89 = h_22_attn_c_proj_bias.broadcast_add(&view_272)?;
        let view_273 = pycandle_core::ops::reshape(&addmm_89, &add_254.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_22_attn_resid_dropout = self.h_22_attn_resid_dropout.forward(&view_273)?;
        py_check!(self.checker, "h.22.attn.resid_dropout", &h_22_attn_resid_dropout);
        let add_255 = h_22_attn_resid_dropout.broadcast_add(&add_249)?;
        let h_22_ln_2 = self.h_22_ln_2.forward(&add_255)?;
        py_check!(self.checker, "h.22.ln_2", &h_22_ln_2);
        let size_279 = h_22_ln_2.dims().to_vec();
        let getitem_305 = size_279[..size_279.len() - 1].to_vec();
        let add_256 = { let mut v: Vec<isize> = getitem_305.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_22_mlp_c_fc_bias = self.h_22_mlp_c_fc.bias().unwrap().clone();
        let size_280 = pycandle_core::ops::dim(&h_22_ln_2, -1 as isize)?;
        let view_274 = pycandle_core::ops::reshape(&h_22_ln_2, &vec![-1isize, size_280 as isize])?;
        let h_22_mlp_c_fc_weight = self.h_22_mlp_c_fc.weight().clone();
        let addmm_90 = h_22_mlp_c_fc_bias.broadcast_add(&view_274)?;
        let view_275 = pycandle_core::ops::reshape(&addmm_90, &add_256.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_88 = view_275.affine(0.5f64, 0.0f64)?;
        let pow_23 = view_275.powf(3.0)?;
        let mul_89 = pow_23.affine(0.044715f64, 0.0f64)?;
        let add_257 = view_275.broadcast_add(&mul_89)?;
        let mul_90 = add_257.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_22 = mul_90.tanh()?;
        let add_258 = tanh_22.affine(1.0, 0.0)?;
        let mul_91 = mul_88.broadcast_mul(&add_258)?;
        let size_281 = mul_91.dims().to_vec();
        let getitem_306 = size_281[..size_281.len() - 1].to_vec();
        let add_259 = { let mut v: Vec<isize> = getitem_306.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_22_mlp_c_proj_bias = self.h_22_mlp_c_proj.bias().unwrap().clone();
        let size_282 = pycandle_core::ops::dim(&mul_91, -1 as isize)?;
        let view_276 = pycandle_core::ops::reshape(&mul_91, &vec![-1isize, size_282 as isize])?;
        let h_22_mlp_c_proj_weight = self.h_22_mlp_c_proj.weight().clone();
        let addmm_91 = h_22_mlp_c_proj_bias.broadcast_add(&view_276)?;
        let view_277 = pycandle_core::ops::reshape(&addmm_91, &add_259.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_22_mlp_dropout = self.h_22_mlp_dropout.forward(&view_277)?;
        py_check!(self.checker, "h.22.mlp.dropout", &h_22_mlp_dropout);
        let add_260 = add_255.broadcast_add(&h_22_mlp_dropout)?;
        let h_23_ln_1 = self.h_23_ln_1.forward(&add_260)?;
        py_check!(self.checker, "h.23.ln_1", &h_23_ln_1);
        let size_283 = h_23_ln_1.dims().to_vec();
        let getitem_307 = size_283[0].clone();
        let getitem_308 = size_283[1].clone();
        let getitem_309 = size_283[2].clone();
        let size_284 = h_23_ln_1.dims().to_vec();
        let getitem_310 = size_284[..size_284.len() - 1].to_vec();
        let add_261 = { let mut v: Vec<isize> = getitem_310.iter().map(|&x| x as isize).collect(); v.extend(vec![3072].iter().map(|&x| x as isize)); v };
        let h_23_attn_c_attn_bias = self.h_23_attn_c_attn.bias().unwrap().clone();
        let size_285 = pycandle_core::ops::dim(&h_23_ln_1, -1 as isize)?;
        let view_278 = pycandle_core::ops::reshape(&h_23_ln_1, &vec![-1isize, size_285 as isize])?;
        let h_23_attn_c_attn_weight = self.h_23_attn_c_attn.weight().clone();
        let addmm_92 = h_23_attn_c_attn_bias.broadcast_add(&view_278)?;
        let view_279 = pycandle_core::ops::reshape(&addmm_92, &add_261.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let split_23 = pycandle_core::ops::split(&view_279, 1024 as usize, 0 as usize)?;
        let getitem_311 = split_23[0].clone();
        let getitem_312 = split_23[1].clone();
        let getitem_313 = split_23[2].clone();
        let size_286 = getitem_311.dims().to_vec();
        let getitem_314 = size_286[..size_286.len() - 1].to_vec();
        let add_262 = { let mut v: Vec<isize> = getitem_314.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_280 = pycandle_core::ops::reshape(&getitem_311, &add_262.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_69 = view_280.permute(0)?;
        let size_287 = getitem_312.dims().to_vec();
        let getitem_315 = size_287[..size_287.len() - 1].to_vec();
        let add_263 = { let mut v: Vec<isize> = getitem_315.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_281 = pycandle_core::ops::reshape(&getitem_312, &add_263.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_70 = view_281.permute(0)?;
        let size_288 = getitem_313.dims().to_vec();
        let getitem_316 = size_288[..size_288.len() - 1].to_vec();
        let add_264 = { let mut v: Vec<isize> = getitem_316.iter().map(|&x| x as isize).collect(); v.extend(vec![16, 64].iter().map(|&x| x as isize)); v };
        let view_282 = pycandle_core::ops::reshape(&getitem_313, &add_264.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let permute_71 = view_282.permute(0)?;
        let scaled_dot_product_attention_23 = pycandle_core::ops::scaled_dot_product_attention(&permute_69, &permute_70, &permute_71, None, 0.0, false, None)?;
        let transpose_23 = scaled_dot_product_attention_23.transpose(1, 2)?;
        let contiguous_23 = transpose_23.contiguous()?;
        let view_283 = pycandle_core::ops::reshape(&contiguous_23, &vec![getitem_307 as isize, getitem_308 as isize, 1024isize])?;
        let size_289 = view_283.dims().to_vec();
        let getitem_317 = size_289[..size_289.len() - 1].to_vec();
        let add_265 = { let mut v: Vec<isize> = getitem_317.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_23_attn_c_proj_bias = self.h_23_attn_c_proj.bias().unwrap().clone();
        let size_290 = pycandle_core::ops::dim(&view_283, -1 as isize)?;
        let view_284 = pycandle_core::ops::reshape(&view_283, &vec![-1isize, size_290 as isize])?;
        let h_23_attn_c_proj_weight = self.h_23_attn_c_proj.weight().clone();
        let addmm_93 = h_23_attn_c_proj_bias.broadcast_add(&view_284)?;
        let view_285 = pycandle_core::ops::reshape(&addmm_93, &add_265.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_23_attn_resid_dropout = self.h_23_attn_resid_dropout.forward(&view_285)?;
        py_check!(self.checker, "h.23.attn.resid_dropout", &h_23_attn_resid_dropout);
        let add_266 = h_23_attn_resid_dropout.broadcast_add(&add_260)?;
        let h_23_ln_2 = self.h_23_ln_2.forward(&add_266)?;
        py_check!(self.checker, "h.23.ln_2", &h_23_ln_2);
        let size_291 = h_23_ln_2.dims().to_vec();
        let getitem_318 = size_291[..size_291.len() - 1].to_vec();
        let add_267 = { let mut v: Vec<isize> = getitem_318.iter().map(|&x| x as isize).collect(); v.extend(vec![4096].iter().map(|&x| x as isize)); v };
        let h_23_mlp_c_fc_bias = self.h_23_mlp_c_fc.bias().unwrap().clone();
        let size_292 = pycandle_core::ops::dim(&h_23_ln_2, -1 as isize)?;
        let view_286 = pycandle_core::ops::reshape(&h_23_ln_2, &vec![-1isize, size_292 as isize])?;
        let h_23_mlp_c_fc_weight = self.h_23_mlp_c_fc.weight().clone();
        let addmm_94 = h_23_mlp_c_fc_bias.broadcast_add(&view_286)?;
        let view_287 = pycandle_core::ops::reshape(&addmm_94, &add_267.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let mul_92 = view_287.affine(0.5f64, 0.0f64)?;
        let pow_24 = view_287.powf(3.0)?;
        let mul_93 = pow_24.affine(0.044715f64, 0.0f64)?;
        let add_268 = view_287.broadcast_add(&mul_93)?;
        let mul_94 = add_268.affine(0.7978845608028654f64, 0.0f64)?;
        let tanh_23 = mul_94.tanh()?;
        let add_269 = tanh_23.affine(1.0, 0.0)?;
        let mul_95 = mul_92.broadcast_mul(&add_269)?;
        let size_293 = mul_95.dims().to_vec();
        let getitem_319 = size_293[..size_293.len() - 1].to_vec();
        let add_270 = { let mut v: Vec<isize> = getitem_319.iter().map(|&x| x as isize).collect(); v.extend(vec![1024].iter().map(|&x| x as isize)); v };
        let h_23_mlp_c_proj_bias = self.h_23_mlp_c_proj.bias().unwrap().clone();
        let size_294 = pycandle_core::ops::dim(&mul_95, -1 as isize)?;
        let view_288 = pycandle_core::ops::reshape(&mul_95, &vec![-1isize, size_294 as isize])?;
        let h_23_mlp_c_proj_weight = self.h_23_mlp_c_proj.weight().clone();
        let addmm_95 = h_23_mlp_c_proj_bias.broadcast_add(&view_288)?;
        let view_289 = pycandle_core::ops::reshape(&addmm_95, &add_270.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        let h_23_mlp_dropout = self.h_23_mlp_dropout.forward(&view_289)?;
        py_check!(self.checker, "h.23.mlp.dropout", &h_23_mlp_dropout);
        let add_271 = add_266.broadcast_add(&h_23_mlp_dropout)?;
        let ln_f = self.ln_f.forward(&add_271)?;
        py_check!(self.checker, "ln_f", &ln_f);
        let view_290 = pycandle_core::ops::reshape(&ln_f, &add_7.iter().map(|&x| x as isize).collect::<Vec<_>>())?;
        Ok(view_290)
    }
}
