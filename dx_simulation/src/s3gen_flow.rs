use candle_core::{Result, Tensor};
use candle_nn::{Module, VarBuilder};
use pycandle_core::{PyChecker, py_check, layers::*};

pub struct Config {
    pub hidden_dim: usize, // 512
    pub vocab_size: usize, // 6561
}
pub struct S3GenFlow {
    pub decoder_estimator_down_blocks_0_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_down_blocks_0_0_block1_block_1: Transpose,
    pub decoder_estimator_down_blocks_0_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_0_block1_block_3: Transpose,
    pub decoder_estimator_down_blocks_0_0_block1_block_4: Mish,
    pub decoder_estimator_down_blocks_0_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_down_blocks_0_0_block2_block_1: Transpose,
    pub decoder_estimator_down_blocks_0_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_0_block2_block_3: Transpose,
    pub decoder_estimator_down_blocks_0_0_block2_block_4: Mish,
    pub decoder_estimator_down_blocks_0_0_mlp_0: Mish,
    pub decoder_estimator_down_blocks_0_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_down_blocks_0_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_ff_net_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_ff_net_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_ff_net_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_ff_net_1: Dropout,
    pub decoder_estimator_down_blocks_0_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_down_blocks_0_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_down_blocks_0_2: CausalConv1d,
    pub decoder_estimator_final_block_block_0: CausalConv1d,
    pub decoder_estimator_final_block_block_1: Transpose,
    pub decoder_estimator_final_block_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_final_block_block_3: Transpose,
    pub decoder_estimator_final_block_block_4: Mish,
    pub decoder_estimator_final_proj: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_0_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_0_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_0_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_0_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_0_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_0_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_0_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_0_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_0_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_0_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_0_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_0_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_0_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_0_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_1_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_1_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_1_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_1_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_1_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_1_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_1_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_1_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_1_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_1_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_1_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_1_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_1_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_10_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_10_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_10_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_10_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_10_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_10_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_10_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_10_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_10_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_10_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_10_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_10_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_10_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_11_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_11_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_11_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_11_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_11_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_11_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_11_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_11_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_11_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_11_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_11_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_11_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_11_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_2_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_2_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_2_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_2_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_2_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_2_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_2_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_2_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_2_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_2_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_2_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_2_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_2_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_3_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_3_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_3_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_3_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_3_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_3_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_3_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_3_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_3_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_3_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_3_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_3_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_3_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_4_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_4_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_4_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_4_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_4_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_4_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_4_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_4_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_4_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_4_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_4_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_4_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_4_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_5_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_5_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_5_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_5_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_5_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_5_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_5_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_5_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_5_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_5_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_5_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_5_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_5_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_6_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_6_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_6_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_6_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_6_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_6_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_6_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_6_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_6_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_6_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_6_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_6_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_6_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_7_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_7_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_7_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_7_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_7_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_7_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_7_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_7_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_7_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_7_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_7_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_7_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_7_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_8_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_8_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_8_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_8_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_8_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_8_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_8_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_8_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_8_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_8_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_8_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_8_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_8_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_9_0_block1_block_1: Transpose,
    pub decoder_estimator_mid_blocks_9_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_0_block1_block_3: Transpose,
    pub decoder_estimator_mid_blocks_9_0_block1_block_4: Mish,
    pub decoder_estimator_mid_blocks_9_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_mid_blocks_9_0_block2_block_1: Transpose,
    pub decoder_estimator_mid_blocks_9_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_0_block2_block_3: Transpose,
    pub decoder_estimator_mid_blocks_9_0_block2_block_4: Mish,
    pub decoder_estimator_mid_blocks_9_0_mlp_0: Mish,
    pub decoder_estimator_mid_blocks_9_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_mid_blocks_9_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_ff_net_1: Dropout,
    pub decoder_estimator_mid_blocks_9_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_mid_blocks_9_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_mid_blocks_9_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_time_embeddings: SinusoidalPosEmb,
    pub decoder_estimator_time_mlp_act: SiLU,
    pub decoder_estimator_time_mlp_linear_1: candle_nn::Linear,
    pub decoder_estimator_time_mlp_linear_2: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_0_block1_block_0: CausalConv1d,
    pub decoder_estimator_up_blocks_0_0_block1_block_1: Transpose,
    pub decoder_estimator_up_blocks_0_0_block1_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_0_block1_block_3: Transpose,
    pub decoder_estimator_up_blocks_0_0_block1_block_4: Mish,
    pub decoder_estimator_up_blocks_0_0_block2_block_0: CausalConv1d,
    pub decoder_estimator_up_blocks_0_0_block2_block_1: Transpose,
    pub decoder_estimator_up_blocks_0_0_block2_block_2: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_0_block2_block_3: Transpose,
    pub decoder_estimator_up_blocks_0_0_block2_block_4: Mish,
    pub decoder_estimator_up_blocks_0_0_mlp_0: Mish,
    pub decoder_estimator_up_blocks_0_0_mlp_1: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_0_res_conv: candle_nn::Conv1d,
    pub decoder_estimator_up_blocks_0_1_0_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_attn1_to_out_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_0_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_ff_net_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_0_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_0_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_0_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_1_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_attn1_to_out_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_1_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_ff_net_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_1_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_1_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_1_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_2_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_attn1_to_out_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_2_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_ff_net_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_2_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_2_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_2_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_3_attn1_to_k: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_attn1_to_out_0: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_attn1_to_out_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_3_attn1_to_q: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_attn1_to_v: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_ff_net_0_proj: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_ff_net_1: Dropout,
    pub decoder_estimator_up_blocks_0_1_3_ff_net_2: candle_nn::Linear,
    pub decoder_estimator_up_blocks_0_1_3_norm1: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_1_3_norm3: candle_nn::LayerNorm,
    pub decoder_estimator_up_blocks_0_2: CausalConv1d,
    pub encoder_after_norm: candle_nn::LayerNorm,
    pub encoder_embed_out_0: candle_nn::Linear,
    pub encoder_embed_out_1: candle_nn::LayerNorm,
    pub encoder_embed_out_2: Dropout,
    pub encoder_embed_pos_enc_dropout: Dropout,
    pub encoder_embed_pos_enc_dropout_1: Dropout,
    pub encoder_encoders_0_dropout: Dropout,
    pub encoder_encoders_0_dropout_1: Dropout,
    pub encoder_encoders_0_feed_forward_activation: SiLU,
    pub encoder_encoders_0_feed_forward_activation_1: SiLU,
    pub encoder_encoders_0_feed_forward_activation_2: SiLU,
    pub encoder_encoders_0_feed_forward_activation_3: SiLU,
    pub encoder_encoders_0_feed_forward_activation_4: SiLU,
    pub encoder_encoders_0_feed_forward_activation_5: SiLU,
    pub encoder_encoders_0_feed_forward_activation_6: SiLU,
    pub encoder_encoders_0_feed_forward_activation_7: SiLU,
    pub encoder_encoders_0_feed_forward_activation_8: SiLU,
    pub encoder_encoders_0_feed_forward_activation_9: SiLU,
    pub encoder_encoders_0_feed_forward_dropout: Dropout,
    pub encoder_encoders_0_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_0_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_0_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_0_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_0_self_attn_dropout: Dropout,
    pub encoder_encoders_0_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_0_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_0_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_0_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_0_self_attn_linear_v: candle_nn::Linear,
    pub encoder_encoders_1_dropout: Dropout,
    pub encoder_encoders_1_dropout_1: Dropout,
    pub encoder_encoders_1_feed_forward_dropout: Dropout,
    pub encoder_encoders_1_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_1_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_1_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_1_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_1_self_attn_dropout: Dropout,
    pub encoder_encoders_1_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_1_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_1_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_1_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_1_self_attn_linear_v: candle_nn::Linear,
    pub encoder_encoders_2_dropout: Dropout,
    pub encoder_encoders_2_dropout_1: Dropout,
    pub encoder_encoders_2_feed_forward_dropout: Dropout,
    pub encoder_encoders_2_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_2_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_2_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_2_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_2_self_attn_dropout: Dropout,
    pub encoder_encoders_2_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_2_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_2_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_2_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_2_self_attn_linear_v: candle_nn::Linear,
    pub encoder_encoders_3_dropout: Dropout,
    pub encoder_encoders_3_dropout_1: Dropout,
    pub encoder_encoders_3_feed_forward_dropout: Dropout,
    pub encoder_encoders_3_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_3_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_3_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_3_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_3_self_attn_dropout: Dropout,
    pub encoder_encoders_3_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_3_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_3_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_3_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_3_self_attn_linear_v: candle_nn::Linear,
    pub encoder_encoders_4_dropout: Dropout,
    pub encoder_encoders_4_dropout_1: Dropout,
    pub encoder_encoders_4_feed_forward_dropout: Dropout,
    pub encoder_encoders_4_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_4_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_4_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_4_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_4_self_attn_dropout: Dropout,
    pub encoder_encoders_4_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_4_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_4_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_4_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_4_self_attn_linear_v: candle_nn::Linear,
    pub encoder_encoders_5_dropout: Dropout,
    pub encoder_encoders_5_dropout_1: Dropout,
    pub encoder_encoders_5_feed_forward_dropout: Dropout,
    pub encoder_encoders_5_feed_forward_w_1: candle_nn::Linear,
    pub encoder_encoders_5_feed_forward_w_2: candle_nn::Linear,
    pub encoder_encoders_5_norm_ff: candle_nn::LayerNorm,
    pub encoder_encoders_5_norm_mha: candle_nn::LayerNorm,
    pub encoder_encoders_5_self_attn_dropout: Dropout,
    pub encoder_encoders_5_self_attn_linear_k: candle_nn::Linear,
    pub encoder_encoders_5_self_attn_linear_out: candle_nn::Linear,
    pub encoder_encoders_5_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_encoders_5_self_attn_linear_q: candle_nn::Linear,
    pub encoder_encoders_5_self_attn_linear_v: candle_nn::Linear,
    pub encoder_pre_lookahead_layer_conv1: candle_nn::Conv1d,
    pub encoder_pre_lookahead_layer_conv2: candle_nn::Conv1d,
    pub encoder_up_embed_out_0: candle_nn::Linear,
    pub encoder_up_embed_out_1: candle_nn::LayerNorm,
    pub encoder_up_embed_out_2: Dropout,
    pub encoder_up_embed_pos_enc_dropout: Dropout,
    pub encoder_up_embed_pos_enc_dropout_1: Dropout,
    pub encoder_up_encoders_0_dropout: Dropout,
    pub encoder_up_encoders_0_dropout_1: Dropout,
    pub encoder_up_encoders_0_feed_forward_dropout: Dropout,
    pub encoder_up_encoders_0_feed_forward_w_1: candle_nn::Linear,
    pub encoder_up_encoders_0_feed_forward_w_2: candle_nn::Linear,
    pub encoder_up_encoders_0_norm_ff: candle_nn::LayerNorm,
    pub encoder_up_encoders_0_norm_mha: candle_nn::LayerNorm,
    pub encoder_up_encoders_0_self_attn_dropout: Dropout,
    pub encoder_up_encoders_0_self_attn_linear_k: candle_nn::Linear,
    pub encoder_up_encoders_0_self_attn_linear_out: candle_nn::Linear,
    pub encoder_up_encoders_0_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_up_encoders_0_self_attn_linear_q: candle_nn::Linear,
    pub encoder_up_encoders_0_self_attn_linear_v: candle_nn::Linear,
    pub encoder_up_encoders_1_dropout: Dropout,
    pub encoder_up_encoders_1_dropout_1: Dropout,
    pub encoder_up_encoders_1_feed_forward_dropout: Dropout,
    pub encoder_up_encoders_1_feed_forward_w_1: candle_nn::Linear,
    pub encoder_up_encoders_1_feed_forward_w_2: candle_nn::Linear,
    pub encoder_up_encoders_1_norm_ff: candle_nn::LayerNorm,
    pub encoder_up_encoders_1_norm_mha: candle_nn::LayerNorm,
    pub encoder_up_encoders_1_self_attn_dropout: Dropout,
    pub encoder_up_encoders_1_self_attn_linear_k: candle_nn::Linear,
    pub encoder_up_encoders_1_self_attn_linear_out: candle_nn::Linear,
    pub encoder_up_encoders_1_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_up_encoders_1_self_attn_linear_q: candle_nn::Linear,
    pub encoder_up_encoders_1_self_attn_linear_v: candle_nn::Linear,
    pub encoder_up_encoders_2_dropout: Dropout,
    pub encoder_up_encoders_2_dropout_1: Dropout,
    pub encoder_up_encoders_2_feed_forward_dropout: Dropout,
    pub encoder_up_encoders_2_feed_forward_w_1: candle_nn::Linear,
    pub encoder_up_encoders_2_feed_forward_w_2: candle_nn::Linear,
    pub encoder_up_encoders_2_norm_ff: candle_nn::LayerNorm,
    pub encoder_up_encoders_2_norm_mha: candle_nn::LayerNorm,
    pub encoder_up_encoders_2_self_attn_dropout: Dropout,
    pub encoder_up_encoders_2_self_attn_linear_k: candle_nn::Linear,
    pub encoder_up_encoders_2_self_attn_linear_out: candle_nn::Linear,
    pub encoder_up_encoders_2_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_up_encoders_2_self_attn_linear_q: candle_nn::Linear,
    pub encoder_up_encoders_2_self_attn_linear_v: candle_nn::Linear,
    pub encoder_up_encoders_3_dropout: Dropout,
    pub encoder_up_encoders_3_dropout_1: Dropout,
    pub encoder_up_encoders_3_feed_forward_dropout: Dropout,
    pub encoder_up_encoders_3_feed_forward_w_1: candle_nn::Linear,
    pub encoder_up_encoders_3_feed_forward_w_2: candle_nn::Linear,
    pub encoder_up_encoders_3_norm_ff: candle_nn::LayerNorm,
    pub encoder_up_encoders_3_norm_mha: candle_nn::LayerNorm,
    pub encoder_up_encoders_3_self_attn_dropout: Dropout,
    pub encoder_up_encoders_3_self_attn_linear_k: candle_nn::Linear,
    pub encoder_up_encoders_3_self_attn_linear_out: candle_nn::Linear,
    pub encoder_up_encoders_3_self_attn_linear_pos: candle_nn::Linear,
    pub encoder_up_encoders_3_self_attn_linear_q: candle_nn::Linear,
    pub encoder_up_encoders_3_self_attn_linear_v: candle_nn::Linear,
    pub encoder_up_layer_conv: candle_nn::Conv1d,
    pub encoder_proj: candle_nn::Linear,
    pub input_embedding: candle_nn::Embedding,
    pub spk_embed_affine_layer: candle_nn::Linear,
    pub checker: Option<PyChecker>,
}

impl S3GenFlow {
    #[allow(unused_variables)]
    pub fn load(config: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let decoder_estimator_down_blocks_0_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.down_blocks.0.0.block1.block.0"), 320, 256, 3, 1, true)?;
        let decoder_estimator_down_blocks_0_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_down_blocks_0_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.0.block1.block.2"))?;
        let decoder_estimator_down_blocks_0_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_down_blocks_0_0_block1_block_4 = Mish;
        let decoder_estimator_down_blocks_0_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.down_blocks.0.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_down_blocks_0_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_down_blocks_0_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.0.block2.block.2"))?;
        let decoder_estimator_down_blocks_0_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_down_blocks_0_0_block2_block_4 = Mish;
        let decoder_estimator_down_blocks_0_0_mlp_0 = Mish;
        let decoder_estimator_down_blocks_0_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.down_blocks.0.0.mlp.1"))?;
        let decoder_estimator_down_blocks_0_0_res_conv = candle_nn::conv1d(320, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.0.res_conv"))?;
        let decoder_estimator_down_blocks_0_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.0.attn1.to_k"))?;
        let decoder_estimator_down_blocks_0_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.down_blocks.0.1.0.attn1.to_out.0"))?;
        let decoder_estimator_down_blocks_0_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.0.attn1.to_q"))?;
        let decoder_estimator_down_blocks_0_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.0.attn1.to_v"))?;
        let decoder_estimator_down_blocks_0_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.down_blocks.0.1.0.ff.net.0.proj"))?;
        let decoder_estimator_down_blocks_0_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.down_blocks.0.1.0.ff.net.2"))?;
        let decoder_estimator_down_blocks_0_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.0.norm1"))?;
        let decoder_estimator_down_blocks_0_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.0.norm3"))?;
        let decoder_estimator_down_blocks_0_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.1.attn1.to_k"))?;
        let decoder_estimator_down_blocks_0_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.down_blocks.0.1.1.attn1.to_out.0"))?;
        let decoder_estimator_down_blocks_0_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.1.attn1.to_q"))?;
        let decoder_estimator_down_blocks_0_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.1.attn1.to_v"))?;
        let decoder_estimator_down_blocks_0_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.down_blocks.0.1.1.ff.net.0.proj"))?;
        let decoder_estimator_down_blocks_0_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.down_blocks.0.1.1.ff.net.2"))?;
        let decoder_estimator_down_blocks_0_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.1.norm1"))?;
        let decoder_estimator_down_blocks_0_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.1.norm3"))?;
        let decoder_estimator_down_blocks_0_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.2.attn1.to_k"))?;
        let decoder_estimator_down_blocks_0_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.down_blocks.0.1.2.attn1.to_out.0"))?;
        let decoder_estimator_down_blocks_0_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.2.attn1.to_q"))?;
        let decoder_estimator_down_blocks_0_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.2.attn1.to_v"))?;
        let decoder_estimator_down_blocks_0_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.down_blocks.0.1.2.ff.net.0.proj"))?;
        let decoder_estimator_down_blocks_0_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.down_blocks.0.1.2.ff.net.2"))?;
        let decoder_estimator_down_blocks_0_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.2.norm1"))?;
        let decoder_estimator_down_blocks_0_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.2.norm3"))?;
        let decoder_estimator_down_blocks_0_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.3.attn1.to_k"))?;
        let decoder_estimator_down_blocks_0_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.down_blocks.0.1.3.attn1.to_out.0"))?;
        let decoder_estimator_down_blocks_0_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.3.attn1.to_q"))?;
        let decoder_estimator_down_blocks_0_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.down_blocks.0.1.3.attn1.to_v"))?;
        let decoder_estimator_down_blocks_0_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.down_blocks.0.1.3.ff.net.0.proj"))?;
        let decoder_estimator_down_blocks_0_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_down_blocks_0_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.down_blocks.0.1.3.ff.net.2"))?;
        let decoder_estimator_down_blocks_0_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.3.norm1"))?;
        let decoder_estimator_down_blocks_0_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.down_blocks.0.1.3.norm3"))?;
        let decoder_estimator_down_blocks_0_2 = CausalConv1d::load(vb.pp("decoder.estimator.down_blocks.0.2"), 256, 256, 3, 1, true)?;
        let decoder_estimator_final_block_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.final_block.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_final_block_block_1 = Transpose::new(1, 2);
        let decoder_estimator_final_block_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.final_block.block.2"))?;
        let decoder_estimator_final_block_block_3 = Transpose::new(1, 2);
        let decoder_estimator_final_block_block_4 = Mish;
        let decoder_estimator_final_proj = candle_nn::conv1d(256, 80, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.final_proj"))?;
        let decoder_estimator_mid_blocks_0_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.0.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_0_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_0_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_0_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_0_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_0_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.0.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_0_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_0_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_0_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_0_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_0_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_0_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.0.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_0_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.0.res_conv"))?;
        let decoder_estimator_mid_blocks_0_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.0.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_0_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_0_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.0.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_0_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.0.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_0_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_0_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_0_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.0.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_0_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_0_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.0.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_0_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.0.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_0_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_0_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_0_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.0.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_0_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_0_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.0.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_0_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.0.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_0_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_0_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_0_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.0.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_0_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_0_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.0.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.0.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_0_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_0_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.0.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_0_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_0_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.0.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_1_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.1.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_1_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_1_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_1_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_1_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_1_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.1.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_1_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_1_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_1_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_1_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_1_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_1_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.1.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_1_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.0.res_conv"))?;
        let decoder_estimator_mid_blocks_1_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.1.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_1_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_1_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.1.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_1_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.1.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_1_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_1_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_1_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.1.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_1_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_1_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.1.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_1_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.1.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_1_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_1_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_1_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.1.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_1_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_1_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.1.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_1_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.1.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_1_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_1_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_1_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.1.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_1_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_1_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.1.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.1.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_1_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_1_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.1.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_1_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_1_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.1.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_10_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.10.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_10_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_10_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_10_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_10_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_10_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.10.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_10_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_10_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_10_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_10_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_10_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_10_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.10.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_10_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.0.res_conv"))?;
        let decoder_estimator_mid_blocks_10_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.10.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_10_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_10_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.10.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_10_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.10.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_10_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_10_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_10_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.10.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_10_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_10_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.10.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_10_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.10.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_10_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_10_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_10_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.10.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_10_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_10_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.10.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_10_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.10.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_10_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_10_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_10_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.10.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_10_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_10_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.10.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.10.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_10_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_10_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.10.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_10_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_10_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.10.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_11_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.11.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_11_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_11_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_11_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_11_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_11_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.11.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_11_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_11_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_11_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_11_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_11_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_11_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.11.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_11_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.0.res_conv"))?;
        let decoder_estimator_mid_blocks_11_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.11.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_11_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_11_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.11.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_11_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.11.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_11_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_11_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_11_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.11.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_11_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_11_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.11.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_11_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.11.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_11_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_11_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_11_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.11.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_11_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_11_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.11.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_11_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.11.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_11_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_11_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_11_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.11.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_11_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_11_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.11.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.11.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_11_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_11_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.11.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_11_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_11_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.11.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_2_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.2.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_2_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_2_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_2_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_2_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_2_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.2.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_2_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_2_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_2_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_2_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_2_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_2_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.2.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_2_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.0.res_conv"))?;
        let decoder_estimator_mid_blocks_2_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.2.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_2_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_2_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.2.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_2_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.2.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_2_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_2_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_2_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.2.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_2_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_2_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.2.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_2_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.2.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_2_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_2_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_2_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.2.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_2_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_2_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.2.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_2_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.2.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_2_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_2_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_2_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.2.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_2_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_2_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.2.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.2.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_2_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_2_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.2.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_2_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_2_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.2.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_3_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.3.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_3_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_3_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_3_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_3_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_3_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.3.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_3_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_3_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_3_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_3_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_3_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_3_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.3.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_3_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.0.res_conv"))?;
        let decoder_estimator_mid_blocks_3_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.3.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_3_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_3_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.3.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_3_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.3.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_3_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_3_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_3_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.3.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_3_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_3_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.3.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_3_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.3.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_3_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_3_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_3_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.3.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_3_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_3_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.3.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_3_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.3.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_3_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_3_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_3_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.3.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_3_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_3_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.3.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.3.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_3_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_3_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.3.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_3_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_3_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.3.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_4_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.4.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_4_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_4_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_4_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_4_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_4_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.4.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_4_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_4_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_4_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_4_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_4_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_4_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.4.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_4_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.0.res_conv"))?;
        let decoder_estimator_mid_blocks_4_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.4.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_4_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_4_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.4.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_4_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.4.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_4_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_4_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_4_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.4.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_4_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_4_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.4.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_4_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.4.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_4_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_4_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_4_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.4.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_4_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_4_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.4.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_4_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.4.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_4_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_4_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_4_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.4.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_4_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_4_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.4.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.4.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_4_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_4_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.4.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_4_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_4_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.4.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_5_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.5.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_5_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_5_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_5_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_5_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_5_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.5.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_5_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_5_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_5_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_5_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_5_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_5_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.5.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_5_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.0.res_conv"))?;
        let decoder_estimator_mid_blocks_5_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.5.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_5_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_5_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.5.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_5_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.5.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_5_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_5_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_5_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.5.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_5_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_5_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.5.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_5_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.5.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_5_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_5_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_5_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.5.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_5_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_5_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.5.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_5_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.5.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_5_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_5_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_5_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.5.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_5_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_5_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.5.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.5.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_5_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_5_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.5.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_5_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_5_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.5.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_6_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.6.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_6_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_6_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_6_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_6_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_6_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.6.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_6_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_6_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_6_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_6_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_6_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_6_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.6.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_6_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.0.res_conv"))?;
        let decoder_estimator_mid_blocks_6_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.6.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_6_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_6_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.6.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_6_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.6.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_6_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_6_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_6_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.6.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_6_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_6_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.6.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_6_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.6.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_6_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_6_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_6_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.6.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_6_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_6_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.6.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_6_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.6.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_6_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_6_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_6_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.6.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_6_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_6_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.6.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.6.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_6_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_6_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.6.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_6_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_6_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.6.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_7_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.7.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_7_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_7_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_7_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_7_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_7_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.7.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_7_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_7_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_7_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_7_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_7_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_7_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.7.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_7_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.0.res_conv"))?;
        let decoder_estimator_mid_blocks_7_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.7.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_7_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_7_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.7.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_7_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.7.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_7_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_7_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_7_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.7.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_7_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_7_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.7.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_7_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.7.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_7_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_7_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_7_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.7.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_7_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_7_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.7.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_7_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.7.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_7_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_7_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_7_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.7.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_7_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_7_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.7.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.7.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_7_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_7_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.7.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_7_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_7_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.7.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_8_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.8.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_8_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_8_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_8_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_8_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_8_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.8.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_8_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_8_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_8_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_8_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_8_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_8_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.8.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_8_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.0.res_conv"))?;
        let decoder_estimator_mid_blocks_8_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.8.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_8_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_8_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.8.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_8_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.8.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_8_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_8_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_8_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.8.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_8_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_8_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.8.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_8_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.8.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_8_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_8_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_8_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.8.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_8_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_8_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.8.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_8_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.8.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_8_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_8_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_8_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.8.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_8_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_8_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.8.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.8.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_8_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_8_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.8.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_8_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_8_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.8.1.3.norm3"))?;
        let decoder_estimator_mid_blocks_9_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.9.0.block1.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_9_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_9_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.0.block1.block.2"))?;
        let decoder_estimator_mid_blocks_9_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_9_0_block1_block_4 = Mish;
        let decoder_estimator_mid_blocks_9_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.mid_blocks.9.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_mid_blocks_9_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_9_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.0.block2.block.2"))?;
        let decoder_estimator_mid_blocks_9_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_mid_blocks_9_0_block2_block_4 = Mish;
        let decoder_estimator_mid_blocks_9_0_mlp_0 = Mish;
        let decoder_estimator_mid_blocks_9_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.9.0.mlp.1"))?;
        let decoder_estimator_mid_blocks_9_0_res_conv = candle_nn::conv1d(256, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.0.res_conv"))?;
        let decoder_estimator_mid_blocks_9_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.0.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.9.1.0.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_9_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.0.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_9_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.0.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.9.1.0.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_9_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.9.1.0.ff.net.2"))?;
        let decoder_estimator_mid_blocks_9_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.0.norm1"))?;
        let decoder_estimator_mid_blocks_9_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.0.norm3"))?;
        let decoder_estimator_mid_blocks_9_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.1.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.9.1.1.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_9_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.1.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_9_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.1.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.9.1.1.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_9_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.9.1.1.ff.net.2"))?;
        let decoder_estimator_mid_blocks_9_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.1.norm1"))?;
        let decoder_estimator_mid_blocks_9_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.1.norm3"))?;
        let decoder_estimator_mid_blocks_9_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.2.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.9.1.2.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_9_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.2.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_9_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.2.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.9.1.2.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_9_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.9.1.2.ff.net.2"))?;
        let decoder_estimator_mid_blocks_9_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.2.norm1"))?;
        let decoder_estimator_mid_blocks_9_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.2.norm3"))?;
        let decoder_estimator_mid_blocks_9_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.3.attn1.to_k"))?;
        let decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.mid_blocks.9.1.3.attn1.to_out.0"))?;
        let decoder_estimator_mid_blocks_9_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.3.attn1.to_q"))?;
        let decoder_estimator_mid_blocks_9_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.mid_blocks.9.1.3.attn1.to_v"))?;
        let decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.mid_blocks.9.1.3.ff.net.0.proj"))?;
        let decoder_estimator_mid_blocks_9_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_mid_blocks_9_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.mid_blocks.9.1.3.ff.net.2"))?;
        let decoder_estimator_mid_blocks_9_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.3.norm1"))?;
        let decoder_estimator_mid_blocks_9_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.mid_blocks.9.1.3.norm3"))?;
        let decoder_estimator_time_embeddings = SinusoidalPosEmb::new(320);
        let decoder_estimator_time_mlp_act = SiLU;
        let decoder_estimator_time_mlp_linear_1 = candle_nn::linear(320, 1024, vb.pp("decoder.estimator.time_mlp.linear_1"))?;
        let decoder_estimator_time_mlp_linear_2 = { let w = vb.pp("decoder.estimator.time_mlp.linear_2").get((1024, 1024), "weight")?.t()?; let b = Some(vb.pp("decoder.estimator.time_mlp.linear_2").get(1024, "bias")?); candle_nn::Linear::new(w, b) };
        let decoder_estimator_up_blocks_0_0_block1_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.up_blocks.0.0.block1.block.0"), config.hidden_dim, 256, 3, 1, true)?;
        let decoder_estimator_up_blocks_0_0_block1_block_1 = Transpose::new(1, 2);
        let decoder_estimator_up_blocks_0_0_block1_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.0.block1.block.2"))?;
        let decoder_estimator_up_blocks_0_0_block1_block_3 = Transpose::new(1, 2);
        let decoder_estimator_up_blocks_0_0_block1_block_4 = Mish;
        let decoder_estimator_up_blocks_0_0_block2_block_0 = CausalConv1d::load(vb.pp("decoder.estimator.up_blocks.0.0.block2.block.0"), 256, 256, 3, 1, true)?;
        let decoder_estimator_up_blocks_0_0_block2_block_1 = Transpose::new(1, 2);
        let decoder_estimator_up_blocks_0_0_block2_block_2 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.0.block2.block.2"))?;
        let decoder_estimator_up_blocks_0_0_block2_block_3 = Transpose::new(1, 2);
        let decoder_estimator_up_blocks_0_0_block2_block_4 = Mish;
        let decoder_estimator_up_blocks_0_0_mlp_0 = Mish;
        let decoder_estimator_up_blocks_0_0_mlp_1 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.up_blocks.0.0.mlp.1"))?;
        let decoder_estimator_up_blocks_0_0_res_conv = candle_nn::conv1d(config.hidden_dim, 256, 1, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.0.res_conv"))?;
        let decoder_estimator_up_blocks_0_1_0_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.0.attn1.to_k"))?;
        let decoder_estimator_up_blocks_0_1_0_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.up_blocks.0.1.0.attn1.to_out.0"))?;
        let decoder_estimator_up_blocks_0_1_0_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_0_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.0.attn1.to_q"))?;
        let decoder_estimator_up_blocks_0_1_0_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.0.attn1.to_v"))?;
        let decoder_estimator_up_blocks_0_1_0_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.up_blocks.0.1.0.ff.net.0.proj"))?;
        let decoder_estimator_up_blocks_0_1_0_ff_net_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_0_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.up_blocks.0.1.0.ff.net.2"))?;
        let decoder_estimator_up_blocks_0_1_0_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.0.norm1"))?;
        let decoder_estimator_up_blocks_0_1_0_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.0.norm3"))?;
        let decoder_estimator_up_blocks_0_1_1_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.1.attn1.to_k"))?;
        let decoder_estimator_up_blocks_0_1_1_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.up_blocks.0.1.1.attn1.to_out.0"))?;
        let decoder_estimator_up_blocks_0_1_1_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_1_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.1.attn1.to_q"))?;
        let decoder_estimator_up_blocks_0_1_1_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.1.attn1.to_v"))?;
        let decoder_estimator_up_blocks_0_1_1_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.up_blocks.0.1.1.ff.net.0.proj"))?;
        let decoder_estimator_up_blocks_0_1_1_ff_net_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_1_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.up_blocks.0.1.1.ff.net.2"))?;
        let decoder_estimator_up_blocks_0_1_1_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.1.norm1"))?;
        let decoder_estimator_up_blocks_0_1_1_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.1.norm3"))?;
        let decoder_estimator_up_blocks_0_1_2_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.2.attn1.to_k"))?;
        let decoder_estimator_up_blocks_0_1_2_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.up_blocks.0.1.2.attn1.to_out.0"))?;
        let decoder_estimator_up_blocks_0_1_2_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_2_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.2.attn1.to_q"))?;
        let decoder_estimator_up_blocks_0_1_2_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.2.attn1.to_v"))?;
        let decoder_estimator_up_blocks_0_1_2_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.up_blocks.0.1.2.ff.net.0.proj"))?;
        let decoder_estimator_up_blocks_0_1_2_ff_net_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_2_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.up_blocks.0.1.2.ff.net.2"))?;
        let decoder_estimator_up_blocks_0_1_2_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.2.norm1"))?;
        let decoder_estimator_up_blocks_0_1_2_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.2.norm3"))?;
        let decoder_estimator_up_blocks_0_1_3_attn1_to_k = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.3.attn1.to_k"))?;
        let decoder_estimator_up_blocks_0_1_3_attn1_to_out_0 = candle_nn::linear(config.hidden_dim, 256, vb.pp("decoder.estimator.up_blocks.0.1.3.attn1.to_out.0"))?;
        let decoder_estimator_up_blocks_0_1_3_attn1_to_out_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_3_attn1_to_q = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.3.attn1.to_q"))?;
        let decoder_estimator_up_blocks_0_1_3_attn1_to_v = candle_nn::linear_no_bias(256, config.hidden_dim, vb.pp("decoder.estimator.up_blocks.0.1.3.attn1.to_v"))?;
        let decoder_estimator_up_blocks_0_1_3_ff_net_0_proj = candle_nn::linear(256, 1024, vb.pp("decoder.estimator.up_blocks.0.1.3.ff.net.0.proj"))?;
        let decoder_estimator_up_blocks_0_1_3_ff_net_1 = Dropout::new();
        let decoder_estimator_up_blocks_0_1_3_ff_net_2 = candle_nn::linear(1024, 256, vb.pp("decoder.estimator.up_blocks.0.1.3.ff.net.2"))?;
        let decoder_estimator_up_blocks_0_1_3_norm1 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.3.norm1"))?;
        let decoder_estimator_up_blocks_0_1_3_norm3 = candle_nn::layer_norm(256, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("decoder.estimator.up_blocks.0.1.3.norm3"))?;
        let decoder_estimator_up_blocks_0_2 = CausalConv1d::load(vb.pp("decoder.estimator.up_blocks.0.2"), 256, 256, 3, 1, true)?;
        let encoder_after_norm = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("encoder.after_norm"))?;
        let encoder_embed_out_0 = { let w = vb.pp("encoder.embed.out.0").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.embed.out.0").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_embed_out_1 = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("encoder.embed.out.1"))?;
        let encoder_embed_out_2 = Dropout::new();
        let encoder_embed_pos_enc_dropout = Dropout::new();
        let encoder_embed_pos_enc_dropout_1 = Dropout::new();
        let encoder_encoders_0_dropout = Dropout::new();
        let encoder_encoders_0_dropout_1 = Dropout::new();
        let encoder_encoders_0_feed_forward_activation = SiLU;
        let encoder_encoders_0_feed_forward_activation_1 = SiLU;
        let encoder_encoders_0_feed_forward_activation_2 = SiLU;
        let encoder_encoders_0_feed_forward_activation_3 = SiLU;
        let encoder_encoders_0_feed_forward_activation_4 = SiLU;
        let encoder_encoders_0_feed_forward_activation_5 = SiLU;
        let encoder_encoders_0_feed_forward_activation_6 = SiLU;
        let encoder_encoders_0_feed_forward_activation_7 = SiLU;
        let encoder_encoders_0_feed_forward_activation_8 = SiLU;
        let encoder_encoders_0_feed_forward_activation_9 = SiLU;
        let encoder_encoders_0_feed_forward_dropout = Dropout::new();
        let encoder_encoders_0_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.0.feed_forward.w_1"))?;
        let encoder_encoders_0_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.0.feed_forward.w_2"))?;
        let encoder_encoders_0_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.0.norm_ff"))?;
        let encoder_encoders_0_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.0.norm_mha"))?;
        let encoder_encoders_0_self_attn_dropout = Dropout::new();
        let encoder_encoders_0_self_attn_linear_k = { let w = vb.pp("encoder.encoders.0.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.0.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_0_self_attn_linear_out = { let w = vb.pp("encoder.encoders.0.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.0.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_0_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.0.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_0_self_attn_linear_q = { let w = vb.pp("encoder.encoders.0.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.0.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_0_self_attn_linear_v = { let w = vb.pp("encoder.encoders.0.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.0.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_1_dropout = Dropout::new();
        let encoder_encoders_1_dropout_1 = Dropout::new();
        let encoder_encoders_1_feed_forward_dropout = Dropout::new();
        let encoder_encoders_1_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.1.feed_forward.w_1"))?;
        let encoder_encoders_1_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.1.feed_forward.w_2"))?;
        let encoder_encoders_1_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.1.norm_ff"))?;
        let encoder_encoders_1_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.1.norm_mha"))?;
        let encoder_encoders_1_self_attn_dropout = Dropout::new();
        let encoder_encoders_1_self_attn_linear_k = { let w = vb.pp("encoder.encoders.1.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.1.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_1_self_attn_linear_out = { let w = vb.pp("encoder.encoders.1.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.1.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_1_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.1.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_1_self_attn_linear_q = { let w = vb.pp("encoder.encoders.1.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.1.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_1_self_attn_linear_v = { let w = vb.pp("encoder.encoders.1.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.1.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_2_dropout = Dropout::new();
        let encoder_encoders_2_dropout_1 = Dropout::new();
        let encoder_encoders_2_feed_forward_dropout = Dropout::new();
        let encoder_encoders_2_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.2.feed_forward.w_1"))?;
        let encoder_encoders_2_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.2.feed_forward.w_2"))?;
        let encoder_encoders_2_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.2.norm_ff"))?;
        let encoder_encoders_2_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.2.norm_mha"))?;
        let encoder_encoders_2_self_attn_dropout = Dropout::new();
        let encoder_encoders_2_self_attn_linear_k = { let w = vb.pp("encoder.encoders.2.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.2.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_2_self_attn_linear_out = { let w = vb.pp("encoder.encoders.2.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.2.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_2_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.2.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_2_self_attn_linear_q = { let w = vb.pp("encoder.encoders.2.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.2.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_2_self_attn_linear_v = { let w = vb.pp("encoder.encoders.2.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.2.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_3_dropout = Dropout::new();
        let encoder_encoders_3_dropout_1 = Dropout::new();
        let encoder_encoders_3_feed_forward_dropout = Dropout::new();
        let encoder_encoders_3_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.3.feed_forward.w_1"))?;
        let encoder_encoders_3_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.3.feed_forward.w_2"))?;
        let encoder_encoders_3_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.3.norm_ff"))?;
        let encoder_encoders_3_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.3.norm_mha"))?;
        let encoder_encoders_3_self_attn_dropout = Dropout::new();
        let encoder_encoders_3_self_attn_linear_k = { let w = vb.pp("encoder.encoders.3.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.3.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_3_self_attn_linear_out = { let w = vb.pp("encoder.encoders.3.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.3.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_3_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.3.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_3_self_attn_linear_q = { let w = vb.pp("encoder.encoders.3.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.3.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_3_self_attn_linear_v = { let w = vb.pp("encoder.encoders.3.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.3.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_4_dropout = Dropout::new();
        let encoder_encoders_4_dropout_1 = Dropout::new();
        let encoder_encoders_4_feed_forward_dropout = Dropout::new();
        let encoder_encoders_4_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.4.feed_forward.w_1"))?;
        let encoder_encoders_4_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.4.feed_forward.w_2"))?;
        let encoder_encoders_4_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.4.norm_ff"))?;
        let encoder_encoders_4_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.4.norm_mha"))?;
        let encoder_encoders_4_self_attn_dropout = Dropout::new();
        let encoder_encoders_4_self_attn_linear_k = { let w = vb.pp("encoder.encoders.4.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.4.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_4_self_attn_linear_out = { let w = vb.pp("encoder.encoders.4.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.4.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_4_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.4.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_4_self_attn_linear_q = { let w = vb.pp("encoder.encoders.4.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.4.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_4_self_attn_linear_v = { let w = vb.pp("encoder.encoders.4.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.4.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_5_dropout = Dropout::new();
        let encoder_encoders_5_dropout_1 = Dropout::new();
        let encoder_encoders_5_feed_forward_dropout = Dropout::new();
        let encoder_encoders_5_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.encoders.5.feed_forward.w_1"))?;
        let encoder_encoders_5_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.encoders.5.feed_forward.w_2"))?;
        let encoder_encoders_5_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.5.norm_ff"))?;
        let encoder_encoders_5_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.encoders.5.norm_mha"))?;
        let encoder_encoders_5_self_attn_dropout = Dropout::new();
        let encoder_encoders_5_self_attn_linear_k = { let w = vb.pp("encoder.encoders.5.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.5.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_5_self_attn_linear_out = { let w = vb.pp("encoder.encoders.5.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.5.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_5_self_attn_linear_pos = { let w = vb.pp("encoder.encoders.5.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_encoders_5_self_attn_linear_q = { let w = vb.pp("encoder.encoders.5.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.5.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_encoders_5_self_attn_linear_v = { let w = vb.pp("encoder.encoders.5.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.encoders.5.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_pre_lookahead_layer_conv1 = candle_nn::conv1d(config.hidden_dim, config.hidden_dim, 4, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("encoder.pre_lookahead_layer.conv1"))?;
        let encoder_pre_lookahead_layer_conv2 = candle_nn::conv1d(config.hidden_dim, config.hidden_dim, 3, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("encoder.pre_lookahead_layer.conv2"))?;
        let encoder_up_embed_out_0 = { let w = vb.pp("encoder.up_embed.out.0").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_embed.out.0").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_embed_out_1 = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-5, ..Default::default() }, vb.pp("encoder.up_embed.out.1"))?;
        let encoder_up_embed_out_2 = Dropout::new();
        let encoder_up_embed_pos_enc_dropout = Dropout::new();
        let encoder_up_embed_pos_enc_dropout_1 = Dropout::new();
        let encoder_up_encoders_0_dropout = Dropout::new();
        let encoder_up_encoders_0_dropout_1 = Dropout::new();
        let encoder_up_encoders_0_feed_forward_dropout = Dropout::new();
        let encoder_up_encoders_0_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.up_encoders.0.feed_forward.w_1"))?;
        let encoder_up_encoders_0_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.up_encoders.0.feed_forward.w_2"))?;
        let encoder_up_encoders_0_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.0.norm_ff"))?;
        let encoder_up_encoders_0_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.0.norm_mha"))?;
        let encoder_up_encoders_0_self_attn_dropout = Dropout::new();
        let encoder_up_encoders_0_self_attn_linear_k = { let w = vb.pp("encoder.up_encoders.0.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.0.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_0_self_attn_linear_out = { let w = vb.pp("encoder.up_encoders.0.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.0.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_0_self_attn_linear_pos = { let w = vb.pp("encoder.up_encoders.0.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_0_self_attn_linear_q = { let w = vb.pp("encoder.up_encoders.0.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.0.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_0_self_attn_linear_v = { let w = vb.pp("encoder.up_encoders.0.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.0.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_1_dropout = Dropout::new();
        let encoder_up_encoders_1_dropout_1 = Dropout::new();
        let encoder_up_encoders_1_feed_forward_dropout = Dropout::new();
        let encoder_up_encoders_1_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.up_encoders.1.feed_forward.w_1"))?;
        let encoder_up_encoders_1_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.up_encoders.1.feed_forward.w_2"))?;
        let encoder_up_encoders_1_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.1.norm_ff"))?;
        let encoder_up_encoders_1_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.1.norm_mha"))?;
        let encoder_up_encoders_1_self_attn_dropout = Dropout::new();
        let encoder_up_encoders_1_self_attn_linear_k = { let w = vb.pp("encoder.up_encoders.1.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.1.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_1_self_attn_linear_out = { let w = vb.pp("encoder.up_encoders.1.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.1.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_1_self_attn_linear_pos = { let w = vb.pp("encoder.up_encoders.1.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_1_self_attn_linear_q = { let w = vb.pp("encoder.up_encoders.1.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.1.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_1_self_attn_linear_v = { let w = vb.pp("encoder.up_encoders.1.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.1.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_2_dropout = Dropout::new();
        let encoder_up_encoders_2_dropout_1 = Dropout::new();
        let encoder_up_encoders_2_feed_forward_dropout = Dropout::new();
        let encoder_up_encoders_2_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.up_encoders.2.feed_forward.w_1"))?;
        let encoder_up_encoders_2_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.up_encoders.2.feed_forward.w_2"))?;
        let encoder_up_encoders_2_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.2.norm_ff"))?;
        let encoder_up_encoders_2_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.2.norm_mha"))?;
        let encoder_up_encoders_2_self_attn_dropout = Dropout::new();
        let encoder_up_encoders_2_self_attn_linear_k = { let w = vb.pp("encoder.up_encoders.2.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.2.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_2_self_attn_linear_out = { let w = vb.pp("encoder.up_encoders.2.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.2.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_2_self_attn_linear_pos = { let w = vb.pp("encoder.up_encoders.2.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_2_self_attn_linear_q = { let w = vb.pp("encoder.up_encoders.2.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.2.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_2_self_attn_linear_v = { let w = vb.pp("encoder.up_encoders.2.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.2.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_3_dropout = Dropout::new();
        let encoder_up_encoders_3_dropout_1 = Dropout::new();
        let encoder_up_encoders_3_feed_forward_dropout = Dropout::new();
        let encoder_up_encoders_3_feed_forward_w_1 = candle_nn::linear(config.hidden_dim, 2048, vb.pp("encoder.up_encoders.3.feed_forward.w_1"))?;
        let encoder_up_encoders_3_feed_forward_w_2 = candle_nn::linear(2048, config.hidden_dim, vb.pp("encoder.up_encoders.3.feed_forward.w_2"))?;
        let encoder_up_encoders_3_norm_ff = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.3.norm_ff"))?;
        let encoder_up_encoders_3_norm_mha = candle_nn::layer_norm(512, candle_nn::LayerNormConfig { eps: 1.0e-12, ..Default::default() }, vb.pp("encoder.up_encoders.3.norm_mha"))?;
        let encoder_up_encoders_3_self_attn_dropout = Dropout::new();
        let encoder_up_encoders_3_self_attn_linear_k = { let w = vb.pp("encoder.up_encoders.3.self_attn.linear_k").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.3.self_attn.linear_k").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_3_self_attn_linear_out = { let w = vb.pp("encoder.up_encoders.3.self_attn.linear_out").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.3.self_attn.linear_out").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_3_self_attn_linear_pos = { let w = vb.pp("encoder.up_encoders.3.self_attn.linear_pos").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = None; candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_3_self_attn_linear_q = { let w = vb.pp("encoder.up_encoders.3.self_attn.linear_q").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.3.self_attn.linear_q").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_encoders_3_self_attn_linear_v = { let w = vb.pp("encoder.up_encoders.3.self_attn.linear_v").get((config.hidden_dim, config.hidden_dim), "weight")?.t()?; let b = Some(vb.pp("encoder.up_encoders.3.self_attn.linear_v").get(config.hidden_dim, "bias")?); candle_nn::Linear::new(w, b) };
        let encoder_up_layer_conv = candle_nn::conv1d(config.hidden_dim, config.hidden_dim, 5, candle_nn::Conv1dConfig { stride: 1, padding: 0, ..Default::default() }, vb.pp("encoder.up_layer.conv"))?;
        let encoder_proj = candle_nn::linear(config.hidden_dim, 80, vb.pp("encoder_proj"))?;
        let input_embedding = candle_nn::embedding(config.vocab_size, config.hidden_dim, vb.pp("input_embedding"))?;
        let spk_embed_affine_layer = candle_nn::linear(192, 80, vb.pp("spk_embed_affine_layer"))?;
        Ok(Self { decoder_estimator_down_blocks_0_0_block1_block_0, decoder_estimator_down_blocks_0_0_block1_block_1, decoder_estimator_down_blocks_0_0_block1_block_2, decoder_estimator_down_blocks_0_0_block1_block_3, decoder_estimator_down_blocks_0_0_block1_block_4, decoder_estimator_down_blocks_0_0_block2_block_0, decoder_estimator_down_blocks_0_0_block2_block_1, decoder_estimator_down_blocks_0_0_block2_block_2, decoder_estimator_down_blocks_0_0_block2_block_3, decoder_estimator_down_blocks_0_0_block2_block_4, decoder_estimator_down_blocks_0_0_mlp_0, decoder_estimator_down_blocks_0_0_mlp_1, decoder_estimator_down_blocks_0_0_res_conv, decoder_estimator_down_blocks_0_1_0_attn1_to_k, decoder_estimator_down_blocks_0_1_0_attn1_to_out_0, decoder_estimator_down_blocks_0_1_0_attn1_to_out_1, decoder_estimator_down_blocks_0_1_0_attn1_to_q, decoder_estimator_down_blocks_0_1_0_attn1_to_v, decoder_estimator_down_blocks_0_1_0_ff_net_0_proj, decoder_estimator_down_blocks_0_1_0_ff_net_1, decoder_estimator_down_blocks_0_1_0_ff_net_2, decoder_estimator_down_blocks_0_1_0_norm1, decoder_estimator_down_blocks_0_1_0_norm3, decoder_estimator_down_blocks_0_1_1_attn1_to_k, decoder_estimator_down_blocks_0_1_1_attn1_to_out_0, decoder_estimator_down_blocks_0_1_1_attn1_to_out_1, decoder_estimator_down_blocks_0_1_1_attn1_to_q, decoder_estimator_down_blocks_0_1_1_attn1_to_v, decoder_estimator_down_blocks_0_1_1_ff_net_0_proj, decoder_estimator_down_blocks_0_1_1_ff_net_1, decoder_estimator_down_blocks_0_1_1_ff_net_2, decoder_estimator_down_blocks_0_1_1_norm1, decoder_estimator_down_blocks_0_1_1_norm3, decoder_estimator_down_blocks_0_1_2_attn1_to_k, decoder_estimator_down_blocks_0_1_2_attn1_to_out_0, decoder_estimator_down_blocks_0_1_2_attn1_to_out_1, decoder_estimator_down_blocks_0_1_2_attn1_to_q, decoder_estimator_down_blocks_0_1_2_attn1_to_v, decoder_estimator_down_blocks_0_1_2_ff_net_0_proj, decoder_estimator_down_blocks_0_1_2_ff_net_1, decoder_estimator_down_blocks_0_1_2_ff_net_2, decoder_estimator_down_blocks_0_1_2_norm1, decoder_estimator_down_blocks_0_1_2_norm3, decoder_estimator_down_blocks_0_1_3_attn1_to_k, decoder_estimator_down_blocks_0_1_3_attn1_to_out_0, decoder_estimator_down_blocks_0_1_3_attn1_to_out_1, decoder_estimator_down_blocks_0_1_3_attn1_to_q, decoder_estimator_down_blocks_0_1_3_attn1_to_v, decoder_estimator_down_blocks_0_1_3_ff_net_0_proj, decoder_estimator_down_blocks_0_1_3_ff_net_1, decoder_estimator_down_blocks_0_1_3_ff_net_2, decoder_estimator_down_blocks_0_1_3_norm1, decoder_estimator_down_blocks_0_1_3_norm3, decoder_estimator_down_blocks_0_2, decoder_estimator_final_block_block_0, decoder_estimator_final_block_block_1, decoder_estimator_final_block_block_2, decoder_estimator_final_block_block_3, decoder_estimator_final_block_block_4, decoder_estimator_final_proj, decoder_estimator_mid_blocks_0_0_block1_block_0, decoder_estimator_mid_blocks_0_0_block1_block_1, decoder_estimator_mid_blocks_0_0_block1_block_2, decoder_estimator_mid_blocks_0_0_block1_block_3, decoder_estimator_mid_blocks_0_0_block1_block_4, decoder_estimator_mid_blocks_0_0_block2_block_0, decoder_estimator_mid_blocks_0_0_block2_block_1, decoder_estimator_mid_blocks_0_0_block2_block_2, decoder_estimator_mid_blocks_0_0_block2_block_3, decoder_estimator_mid_blocks_0_0_block2_block_4, decoder_estimator_mid_blocks_0_0_mlp_0, decoder_estimator_mid_blocks_0_0_mlp_1, decoder_estimator_mid_blocks_0_0_res_conv, decoder_estimator_mid_blocks_0_1_0_attn1_to_k, decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_0_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_0_1_0_attn1_to_q, decoder_estimator_mid_blocks_0_1_0_attn1_to_v, decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_0_1_0_ff_net_1, decoder_estimator_mid_blocks_0_1_0_ff_net_2, decoder_estimator_mid_blocks_0_1_0_norm1, decoder_estimator_mid_blocks_0_1_0_norm3, decoder_estimator_mid_blocks_0_1_1_attn1_to_k, decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_0_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_0_1_1_attn1_to_q, decoder_estimator_mid_blocks_0_1_1_attn1_to_v, decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_0_1_1_ff_net_1, decoder_estimator_mid_blocks_0_1_1_ff_net_2, decoder_estimator_mid_blocks_0_1_1_norm1, decoder_estimator_mid_blocks_0_1_1_norm3, decoder_estimator_mid_blocks_0_1_2_attn1_to_k, decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_0_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_0_1_2_attn1_to_q, decoder_estimator_mid_blocks_0_1_2_attn1_to_v, decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_0_1_2_ff_net_1, decoder_estimator_mid_blocks_0_1_2_ff_net_2, decoder_estimator_mid_blocks_0_1_2_norm1, decoder_estimator_mid_blocks_0_1_2_norm3, decoder_estimator_mid_blocks_0_1_3_attn1_to_k, decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_0_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_0_1_3_attn1_to_q, decoder_estimator_mid_blocks_0_1_3_attn1_to_v, decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_0_1_3_ff_net_1, decoder_estimator_mid_blocks_0_1_3_ff_net_2, decoder_estimator_mid_blocks_0_1_3_norm1, decoder_estimator_mid_blocks_0_1_3_norm3, decoder_estimator_mid_blocks_1_0_block1_block_0, decoder_estimator_mid_blocks_1_0_block1_block_1, decoder_estimator_mid_blocks_1_0_block1_block_2, decoder_estimator_mid_blocks_1_0_block1_block_3, decoder_estimator_mid_blocks_1_0_block1_block_4, decoder_estimator_mid_blocks_1_0_block2_block_0, decoder_estimator_mid_blocks_1_0_block2_block_1, decoder_estimator_mid_blocks_1_0_block2_block_2, decoder_estimator_mid_blocks_1_0_block2_block_3, decoder_estimator_mid_blocks_1_0_block2_block_4, decoder_estimator_mid_blocks_1_0_mlp_0, decoder_estimator_mid_blocks_1_0_mlp_1, decoder_estimator_mid_blocks_1_0_res_conv, decoder_estimator_mid_blocks_1_1_0_attn1_to_k, decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_1_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_1_1_0_attn1_to_q, decoder_estimator_mid_blocks_1_1_0_attn1_to_v, decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_1_1_0_ff_net_1, decoder_estimator_mid_blocks_1_1_0_ff_net_2, decoder_estimator_mid_blocks_1_1_0_norm1, decoder_estimator_mid_blocks_1_1_0_norm3, decoder_estimator_mid_blocks_1_1_1_attn1_to_k, decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_1_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_1_1_1_attn1_to_q, decoder_estimator_mid_blocks_1_1_1_attn1_to_v, decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_1_1_1_ff_net_1, decoder_estimator_mid_blocks_1_1_1_ff_net_2, decoder_estimator_mid_blocks_1_1_1_norm1, decoder_estimator_mid_blocks_1_1_1_norm3, decoder_estimator_mid_blocks_1_1_2_attn1_to_k, decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_1_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_1_1_2_attn1_to_q, decoder_estimator_mid_blocks_1_1_2_attn1_to_v, decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_1_1_2_ff_net_1, decoder_estimator_mid_blocks_1_1_2_ff_net_2, decoder_estimator_mid_blocks_1_1_2_norm1, decoder_estimator_mid_blocks_1_1_2_norm3, decoder_estimator_mid_blocks_1_1_3_attn1_to_k, decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_1_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_1_1_3_attn1_to_q, decoder_estimator_mid_blocks_1_1_3_attn1_to_v, decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_1_1_3_ff_net_1, decoder_estimator_mid_blocks_1_1_3_ff_net_2, decoder_estimator_mid_blocks_1_1_3_norm1, decoder_estimator_mid_blocks_1_1_3_norm3, decoder_estimator_mid_blocks_10_0_block1_block_0, decoder_estimator_mid_blocks_10_0_block1_block_1, decoder_estimator_mid_blocks_10_0_block1_block_2, decoder_estimator_mid_blocks_10_0_block1_block_3, decoder_estimator_mid_blocks_10_0_block1_block_4, decoder_estimator_mid_blocks_10_0_block2_block_0, decoder_estimator_mid_blocks_10_0_block2_block_1, decoder_estimator_mid_blocks_10_0_block2_block_2, decoder_estimator_mid_blocks_10_0_block2_block_3, decoder_estimator_mid_blocks_10_0_block2_block_4, decoder_estimator_mid_blocks_10_0_mlp_0, decoder_estimator_mid_blocks_10_0_mlp_1, decoder_estimator_mid_blocks_10_0_res_conv, decoder_estimator_mid_blocks_10_1_0_attn1_to_k, decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_10_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_10_1_0_attn1_to_q, decoder_estimator_mid_blocks_10_1_0_attn1_to_v, decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_10_1_0_ff_net_1, decoder_estimator_mid_blocks_10_1_0_ff_net_2, decoder_estimator_mid_blocks_10_1_0_norm1, decoder_estimator_mid_blocks_10_1_0_norm3, decoder_estimator_mid_blocks_10_1_1_attn1_to_k, decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_10_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_10_1_1_attn1_to_q, decoder_estimator_mid_blocks_10_1_1_attn1_to_v, decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_10_1_1_ff_net_1, decoder_estimator_mid_blocks_10_1_1_ff_net_2, decoder_estimator_mid_blocks_10_1_1_norm1, decoder_estimator_mid_blocks_10_1_1_norm3, decoder_estimator_mid_blocks_10_1_2_attn1_to_k, decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_10_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_10_1_2_attn1_to_q, decoder_estimator_mid_blocks_10_1_2_attn1_to_v, decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_10_1_2_ff_net_1, decoder_estimator_mid_blocks_10_1_2_ff_net_2, decoder_estimator_mid_blocks_10_1_2_norm1, decoder_estimator_mid_blocks_10_1_2_norm3, decoder_estimator_mid_blocks_10_1_3_attn1_to_k, decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_10_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_10_1_3_attn1_to_q, decoder_estimator_mid_blocks_10_1_3_attn1_to_v, decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_10_1_3_ff_net_1, decoder_estimator_mid_blocks_10_1_3_ff_net_2, decoder_estimator_mid_blocks_10_1_3_norm1, decoder_estimator_mid_blocks_10_1_3_norm3, decoder_estimator_mid_blocks_11_0_block1_block_0, decoder_estimator_mid_blocks_11_0_block1_block_1, decoder_estimator_mid_blocks_11_0_block1_block_2, decoder_estimator_mid_blocks_11_0_block1_block_3, decoder_estimator_mid_blocks_11_0_block1_block_4, decoder_estimator_mid_blocks_11_0_block2_block_0, decoder_estimator_mid_blocks_11_0_block2_block_1, decoder_estimator_mid_blocks_11_0_block2_block_2, decoder_estimator_mid_blocks_11_0_block2_block_3, decoder_estimator_mid_blocks_11_0_block2_block_4, decoder_estimator_mid_blocks_11_0_mlp_0, decoder_estimator_mid_blocks_11_0_mlp_1, decoder_estimator_mid_blocks_11_0_res_conv, decoder_estimator_mid_blocks_11_1_0_attn1_to_k, decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_11_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_11_1_0_attn1_to_q, decoder_estimator_mid_blocks_11_1_0_attn1_to_v, decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_11_1_0_ff_net_1, decoder_estimator_mid_blocks_11_1_0_ff_net_2, decoder_estimator_mid_blocks_11_1_0_norm1, decoder_estimator_mid_blocks_11_1_0_norm3, decoder_estimator_mid_blocks_11_1_1_attn1_to_k, decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_11_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_11_1_1_attn1_to_q, decoder_estimator_mid_blocks_11_1_1_attn1_to_v, decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_11_1_1_ff_net_1, decoder_estimator_mid_blocks_11_1_1_ff_net_2, decoder_estimator_mid_blocks_11_1_1_norm1, decoder_estimator_mid_blocks_11_1_1_norm3, decoder_estimator_mid_blocks_11_1_2_attn1_to_k, decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_11_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_11_1_2_attn1_to_q, decoder_estimator_mid_blocks_11_1_2_attn1_to_v, decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_11_1_2_ff_net_1, decoder_estimator_mid_blocks_11_1_2_ff_net_2, decoder_estimator_mid_blocks_11_1_2_norm1, decoder_estimator_mid_blocks_11_1_2_norm3, decoder_estimator_mid_blocks_11_1_3_attn1_to_k, decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_11_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_11_1_3_attn1_to_q, decoder_estimator_mid_blocks_11_1_3_attn1_to_v, decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_11_1_3_ff_net_1, decoder_estimator_mid_blocks_11_1_3_ff_net_2, decoder_estimator_mid_blocks_11_1_3_norm1, decoder_estimator_mid_blocks_11_1_3_norm3, decoder_estimator_mid_blocks_2_0_block1_block_0, decoder_estimator_mid_blocks_2_0_block1_block_1, decoder_estimator_mid_blocks_2_0_block1_block_2, decoder_estimator_mid_blocks_2_0_block1_block_3, decoder_estimator_mid_blocks_2_0_block1_block_4, decoder_estimator_mid_blocks_2_0_block2_block_0, decoder_estimator_mid_blocks_2_0_block2_block_1, decoder_estimator_mid_blocks_2_0_block2_block_2, decoder_estimator_mid_blocks_2_0_block2_block_3, decoder_estimator_mid_blocks_2_0_block2_block_4, decoder_estimator_mid_blocks_2_0_mlp_0, decoder_estimator_mid_blocks_2_0_mlp_1, decoder_estimator_mid_blocks_2_0_res_conv, decoder_estimator_mid_blocks_2_1_0_attn1_to_k, decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_2_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_2_1_0_attn1_to_q, decoder_estimator_mid_blocks_2_1_0_attn1_to_v, decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_2_1_0_ff_net_1, decoder_estimator_mid_blocks_2_1_0_ff_net_2, decoder_estimator_mid_blocks_2_1_0_norm1, decoder_estimator_mid_blocks_2_1_0_norm3, decoder_estimator_mid_blocks_2_1_1_attn1_to_k, decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_2_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_2_1_1_attn1_to_q, decoder_estimator_mid_blocks_2_1_1_attn1_to_v, decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_2_1_1_ff_net_1, decoder_estimator_mid_blocks_2_1_1_ff_net_2, decoder_estimator_mid_blocks_2_1_1_norm1, decoder_estimator_mid_blocks_2_1_1_norm3, decoder_estimator_mid_blocks_2_1_2_attn1_to_k, decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_2_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_2_1_2_attn1_to_q, decoder_estimator_mid_blocks_2_1_2_attn1_to_v, decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_2_1_2_ff_net_1, decoder_estimator_mid_blocks_2_1_2_ff_net_2, decoder_estimator_mid_blocks_2_1_2_norm1, decoder_estimator_mid_blocks_2_1_2_norm3, decoder_estimator_mid_blocks_2_1_3_attn1_to_k, decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_2_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_2_1_3_attn1_to_q, decoder_estimator_mid_blocks_2_1_3_attn1_to_v, decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_2_1_3_ff_net_1, decoder_estimator_mid_blocks_2_1_3_ff_net_2, decoder_estimator_mid_blocks_2_1_3_norm1, decoder_estimator_mid_blocks_2_1_3_norm3, decoder_estimator_mid_blocks_3_0_block1_block_0, decoder_estimator_mid_blocks_3_0_block1_block_1, decoder_estimator_mid_blocks_3_0_block1_block_2, decoder_estimator_mid_blocks_3_0_block1_block_3, decoder_estimator_mid_blocks_3_0_block1_block_4, decoder_estimator_mid_blocks_3_0_block2_block_0, decoder_estimator_mid_blocks_3_0_block2_block_1, decoder_estimator_mid_blocks_3_0_block2_block_2, decoder_estimator_mid_blocks_3_0_block2_block_3, decoder_estimator_mid_blocks_3_0_block2_block_4, decoder_estimator_mid_blocks_3_0_mlp_0, decoder_estimator_mid_blocks_3_0_mlp_1, decoder_estimator_mid_blocks_3_0_res_conv, decoder_estimator_mid_blocks_3_1_0_attn1_to_k, decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_3_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_3_1_0_attn1_to_q, decoder_estimator_mid_blocks_3_1_0_attn1_to_v, decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_3_1_0_ff_net_1, decoder_estimator_mid_blocks_3_1_0_ff_net_2, decoder_estimator_mid_blocks_3_1_0_norm1, decoder_estimator_mid_blocks_3_1_0_norm3, decoder_estimator_mid_blocks_3_1_1_attn1_to_k, decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_3_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_3_1_1_attn1_to_q, decoder_estimator_mid_blocks_3_1_1_attn1_to_v, decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_3_1_1_ff_net_1, decoder_estimator_mid_blocks_3_1_1_ff_net_2, decoder_estimator_mid_blocks_3_1_1_norm1, decoder_estimator_mid_blocks_3_1_1_norm3, decoder_estimator_mid_blocks_3_1_2_attn1_to_k, decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_3_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_3_1_2_attn1_to_q, decoder_estimator_mid_blocks_3_1_2_attn1_to_v, decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_3_1_2_ff_net_1, decoder_estimator_mid_blocks_3_1_2_ff_net_2, decoder_estimator_mid_blocks_3_1_2_norm1, decoder_estimator_mid_blocks_3_1_2_norm3, decoder_estimator_mid_blocks_3_1_3_attn1_to_k, decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_3_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_3_1_3_attn1_to_q, decoder_estimator_mid_blocks_3_1_3_attn1_to_v, decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_3_1_3_ff_net_1, decoder_estimator_mid_blocks_3_1_3_ff_net_2, decoder_estimator_mid_blocks_3_1_3_norm1, decoder_estimator_mid_blocks_3_1_3_norm3, decoder_estimator_mid_blocks_4_0_block1_block_0, decoder_estimator_mid_blocks_4_0_block1_block_1, decoder_estimator_mid_blocks_4_0_block1_block_2, decoder_estimator_mid_blocks_4_0_block1_block_3, decoder_estimator_mid_blocks_4_0_block1_block_4, decoder_estimator_mid_blocks_4_0_block2_block_0, decoder_estimator_mid_blocks_4_0_block2_block_1, decoder_estimator_mid_blocks_4_0_block2_block_2, decoder_estimator_mid_blocks_4_0_block2_block_3, decoder_estimator_mid_blocks_4_0_block2_block_4, decoder_estimator_mid_blocks_4_0_mlp_0, decoder_estimator_mid_blocks_4_0_mlp_1, decoder_estimator_mid_blocks_4_0_res_conv, decoder_estimator_mid_blocks_4_1_0_attn1_to_k, decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_4_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_4_1_0_attn1_to_q, decoder_estimator_mid_blocks_4_1_0_attn1_to_v, decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_4_1_0_ff_net_1, decoder_estimator_mid_blocks_4_1_0_ff_net_2, decoder_estimator_mid_blocks_4_1_0_norm1, decoder_estimator_mid_blocks_4_1_0_norm3, decoder_estimator_mid_blocks_4_1_1_attn1_to_k, decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_4_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_4_1_1_attn1_to_q, decoder_estimator_mid_blocks_4_1_1_attn1_to_v, decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_4_1_1_ff_net_1, decoder_estimator_mid_blocks_4_1_1_ff_net_2, decoder_estimator_mid_blocks_4_1_1_norm1, decoder_estimator_mid_blocks_4_1_1_norm3, decoder_estimator_mid_blocks_4_1_2_attn1_to_k, decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_4_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_4_1_2_attn1_to_q, decoder_estimator_mid_blocks_4_1_2_attn1_to_v, decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_4_1_2_ff_net_1, decoder_estimator_mid_blocks_4_1_2_ff_net_2, decoder_estimator_mid_blocks_4_1_2_norm1, decoder_estimator_mid_blocks_4_1_2_norm3, decoder_estimator_mid_blocks_4_1_3_attn1_to_k, decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_4_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_4_1_3_attn1_to_q, decoder_estimator_mid_blocks_4_1_3_attn1_to_v, decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_4_1_3_ff_net_1, decoder_estimator_mid_blocks_4_1_3_ff_net_2, decoder_estimator_mid_blocks_4_1_3_norm1, decoder_estimator_mid_blocks_4_1_3_norm3, decoder_estimator_mid_blocks_5_0_block1_block_0, decoder_estimator_mid_blocks_5_0_block1_block_1, decoder_estimator_mid_blocks_5_0_block1_block_2, decoder_estimator_mid_blocks_5_0_block1_block_3, decoder_estimator_mid_blocks_5_0_block1_block_4, decoder_estimator_mid_blocks_5_0_block2_block_0, decoder_estimator_mid_blocks_5_0_block2_block_1, decoder_estimator_mid_blocks_5_0_block2_block_2, decoder_estimator_mid_blocks_5_0_block2_block_3, decoder_estimator_mid_blocks_5_0_block2_block_4, decoder_estimator_mid_blocks_5_0_mlp_0, decoder_estimator_mid_blocks_5_0_mlp_1, decoder_estimator_mid_blocks_5_0_res_conv, decoder_estimator_mid_blocks_5_1_0_attn1_to_k, decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_5_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_5_1_0_attn1_to_q, decoder_estimator_mid_blocks_5_1_0_attn1_to_v, decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_5_1_0_ff_net_1, decoder_estimator_mid_blocks_5_1_0_ff_net_2, decoder_estimator_mid_blocks_5_1_0_norm1, decoder_estimator_mid_blocks_5_1_0_norm3, decoder_estimator_mid_blocks_5_1_1_attn1_to_k, decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_5_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_5_1_1_attn1_to_q, decoder_estimator_mid_blocks_5_1_1_attn1_to_v, decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_5_1_1_ff_net_1, decoder_estimator_mid_blocks_5_1_1_ff_net_2, decoder_estimator_mid_blocks_5_1_1_norm1, decoder_estimator_mid_blocks_5_1_1_norm3, decoder_estimator_mid_blocks_5_1_2_attn1_to_k, decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_5_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_5_1_2_attn1_to_q, decoder_estimator_mid_blocks_5_1_2_attn1_to_v, decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_5_1_2_ff_net_1, decoder_estimator_mid_blocks_5_1_2_ff_net_2, decoder_estimator_mid_blocks_5_1_2_norm1, decoder_estimator_mid_blocks_5_1_2_norm3, decoder_estimator_mid_blocks_5_1_3_attn1_to_k, decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_5_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_5_1_3_attn1_to_q, decoder_estimator_mid_blocks_5_1_3_attn1_to_v, decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_5_1_3_ff_net_1, decoder_estimator_mid_blocks_5_1_3_ff_net_2, decoder_estimator_mid_blocks_5_1_3_norm1, decoder_estimator_mid_blocks_5_1_3_norm3, decoder_estimator_mid_blocks_6_0_block1_block_0, decoder_estimator_mid_blocks_6_0_block1_block_1, decoder_estimator_mid_blocks_6_0_block1_block_2, decoder_estimator_mid_blocks_6_0_block1_block_3, decoder_estimator_mid_blocks_6_0_block1_block_4, decoder_estimator_mid_blocks_6_0_block2_block_0, decoder_estimator_mid_blocks_6_0_block2_block_1, decoder_estimator_mid_blocks_6_0_block2_block_2, decoder_estimator_mid_blocks_6_0_block2_block_3, decoder_estimator_mid_blocks_6_0_block2_block_4, decoder_estimator_mid_blocks_6_0_mlp_0, decoder_estimator_mid_blocks_6_0_mlp_1, decoder_estimator_mid_blocks_6_0_res_conv, decoder_estimator_mid_blocks_6_1_0_attn1_to_k, decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_6_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_6_1_0_attn1_to_q, decoder_estimator_mid_blocks_6_1_0_attn1_to_v, decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_6_1_0_ff_net_1, decoder_estimator_mid_blocks_6_1_0_ff_net_2, decoder_estimator_mid_blocks_6_1_0_norm1, decoder_estimator_mid_blocks_6_1_0_norm3, decoder_estimator_mid_blocks_6_1_1_attn1_to_k, decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_6_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_6_1_1_attn1_to_q, decoder_estimator_mid_blocks_6_1_1_attn1_to_v, decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_6_1_1_ff_net_1, decoder_estimator_mid_blocks_6_1_1_ff_net_2, decoder_estimator_mid_blocks_6_1_1_norm1, decoder_estimator_mid_blocks_6_1_1_norm3, decoder_estimator_mid_blocks_6_1_2_attn1_to_k, decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_6_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_6_1_2_attn1_to_q, decoder_estimator_mid_blocks_6_1_2_attn1_to_v, decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_6_1_2_ff_net_1, decoder_estimator_mid_blocks_6_1_2_ff_net_2, decoder_estimator_mid_blocks_6_1_2_norm1, decoder_estimator_mid_blocks_6_1_2_norm3, decoder_estimator_mid_blocks_6_1_3_attn1_to_k, decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_6_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_6_1_3_attn1_to_q, decoder_estimator_mid_blocks_6_1_3_attn1_to_v, decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_6_1_3_ff_net_1, decoder_estimator_mid_blocks_6_1_3_ff_net_2, decoder_estimator_mid_blocks_6_1_3_norm1, decoder_estimator_mid_blocks_6_1_3_norm3, decoder_estimator_mid_blocks_7_0_block1_block_0, decoder_estimator_mid_blocks_7_0_block1_block_1, decoder_estimator_mid_blocks_7_0_block1_block_2, decoder_estimator_mid_blocks_7_0_block1_block_3, decoder_estimator_mid_blocks_7_0_block1_block_4, decoder_estimator_mid_blocks_7_0_block2_block_0, decoder_estimator_mid_blocks_7_0_block2_block_1, decoder_estimator_mid_blocks_7_0_block2_block_2, decoder_estimator_mid_blocks_7_0_block2_block_3, decoder_estimator_mid_blocks_7_0_block2_block_4, decoder_estimator_mid_blocks_7_0_mlp_0, decoder_estimator_mid_blocks_7_0_mlp_1, decoder_estimator_mid_blocks_7_0_res_conv, decoder_estimator_mid_blocks_7_1_0_attn1_to_k, decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_7_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_7_1_0_attn1_to_q, decoder_estimator_mid_blocks_7_1_0_attn1_to_v, decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_7_1_0_ff_net_1, decoder_estimator_mid_blocks_7_1_0_ff_net_2, decoder_estimator_mid_blocks_7_1_0_norm1, decoder_estimator_mid_blocks_7_1_0_norm3, decoder_estimator_mid_blocks_7_1_1_attn1_to_k, decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_7_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_7_1_1_attn1_to_q, decoder_estimator_mid_blocks_7_1_1_attn1_to_v, decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_7_1_1_ff_net_1, decoder_estimator_mid_blocks_7_1_1_ff_net_2, decoder_estimator_mid_blocks_7_1_1_norm1, decoder_estimator_mid_blocks_7_1_1_norm3, decoder_estimator_mid_blocks_7_1_2_attn1_to_k, decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_7_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_7_1_2_attn1_to_q, decoder_estimator_mid_blocks_7_1_2_attn1_to_v, decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_7_1_2_ff_net_1, decoder_estimator_mid_blocks_7_1_2_ff_net_2, decoder_estimator_mid_blocks_7_1_2_norm1, decoder_estimator_mid_blocks_7_1_2_norm3, decoder_estimator_mid_blocks_7_1_3_attn1_to_k, decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_7_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_7_1_3_attn1_to_q, decoder_estimator_mid_blocks_7_1_3_attn1_to_v, decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_7_1_3_ff_net_1, decoder_estimator_mid_blocks_7_1_3_ff_net_2, decoder_estimator_mid_blocks_7_1_3_norm1, decoder_estimator_mid_blocks_7_1_3_norm3, decoder_estimator_mid_blocks_8_0_block1_block_0, decoder_estimator_mid_blocks_8_0_block1_block_1, decoder_estimator_mid_blocks_8_0_block1_block_2, decoder_estimator_mid_blocks_8_0_block1_block_3, decoder_estimator_mid_blocks_8_0_block1_block_4, decoder_estimator_mid_blocks_8_0_block2_block_0, decoder_estimator_mid_blocks_8_0_block2_block_1, decoder_estimator_mid_blocks_8_0_block2_block_2, decoder_estimator_mid_blocks_8_0_block2_block_3, decoder_estimator_mid_blocks_8_0_block2_block_4, decoder_estimator_mid_blocks_8_0_mlp_0, decoder_estimator_mid_blocks_8_0_mlp_1, decoder_estimator_mid_blocks_8_0_res_conv, decoder_estimator_mid_blocks_8_1_0_attn1_to_k, decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_8_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_8_1_0_attn1_to_q, decoder_estimator_mid_blocks_8_1_0_attn1_to_v, decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_8_1_0_ff_net_1, decoder_estimator_mid_blocks_8_1_0_ff_net_2, decoder_estimator_mid_blocks_8_1_0_norm1, decoder_estimator_mid_blocks_8_1_0_norm3, decoder_estimator_mid_blocks_8_1_1_attn1_to_k, decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_8_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_8_1_1_attn1_to_q, decoder_estimator_mid_blocks_8_1_1_attn1_to_v, decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_8_1_1_ff_net_1, decoder_estimator_mid_blocks_8_1_1_ff_net_2, decoder_estimator_mid_blocks_8_1_1_norm1, decoder_estimator_mid_blocks_8_1_1_norm3, decoder_estimator_mid_blocks_8_1_2_attn1_to_k, decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_8_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_8_1_2_attn1_to_q, decoder_estimator_mid_blocks_8_1_2_attn1_to_v, decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_8_1_2_ff_net_1, decoder_estimator_mid_blocks_8_1_2_ff_net_2, decoder_estimator_mid_blocks_8_1_2_norm1, decoder_estimator_mid_blocks_8_1_2_norm3, decoder_estimator_mid_blocks_8_1_3_attn1_to_k, decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_8_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_8_1_3_attn1_to_q, decoder_estimator_mid_blocks_8_1_3_attn1_to_v, decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_8_1_3_ff_net_1, decoder_estimator_mid_blocks_8_1_3_ff_net_2, decoder_estimator_mid_blocks_8_1_3_norm1, decoder_estimator_mid_blocks_8_1_3_norm3, decoder_estimator_mid_blocks_9_0_block1_block_0, decoder_estimator_mid_blocks_9_0_block1_block_1, decoder_estimator_mid_blocks_9_0_block1_block_2, decoder_estimator_mid_blocks_9_0_block1_block_3, decoder_estimator_mid_blocks_9_0_block1_block_4, decoder_estimator_mid_blocks_9_0_block2_block_0, decoder_estimator_mid_blocks_9_0_block2_block_1, decoder_estimator_mid_blocks_9_0_block2_block_2, decoder_estimator_mid_blocks_9_0_block2_block_3, decoder_estimator_mid_blocks_9_0_block2_block_4, decoder_estimator_mid_blocks_9_0_mlp_0, decoder_estimator_mid_blocks_9_0_mlp_1, decoder_estimator_mid_blocks_9_0_res_conv, decoder_estimator_mid_blocks_9_1_0_attn1_to_k, decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0, decoder_estimator_mid_blocks_9_1_0_attn1_to_out_1, decoder_estimator_mid_blocks_9_1_0_attn1_to_q, decoder_estimator_mid_blocks_9_1_0_attn1_to_v, decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj, decoder_estimator_mid_blocks_9_1_0_ff_net_1, decoder_estimator_mid_blocks_9_1_0_ff_net_2, decoder_estimator_mid_blocks_9_1_0_norm1, decoder_estimator_mid_blocks_9_1_0_norm3, decoder_estimator_mid_blocks_9_1_1_attn1_to_k, decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0, decoder_estimator_mid_blocks_9_1_1_attn1_to_out_1, decoder_estimator_mid_blocks_9_1_1_attn1_to_q, decoder_estimator_mid_blocks_9_1_1_attn1_to_v, decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj, decoder_estimator_mid_blocks_9_1_1_ff_net_1, decoder_estimator_mid_blocks_9_1_1_ff_net_2, decoder_estimator_mid_blocks_9_1_1_norm1, decoder_estimator_mid_blocks_9_1_1_norm3, decoder_estimator_mid_blocks_9_1_2_attn1_to_k, decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0, decoder_estimator_mid_blocks_9_1_2_attn1_to_out_1, decoder_estimator_mid_blocks_9_1_2_attn1_to_q, decoder_estimator_mid_blocks_9_1_2_attn1_to_v, decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj, decoder_estimator_mid_blocks_9_1_2_ff_net_1, decoder_estimator_mid_blocks_9_1_2_ff_net_2, decoder_estimator_mid_blocks_9_1_2_norm1, decoder_estimator_mid_blocks_9_1_2_norm3, decoder_estimator_mid_blocks_9_1_3_attn1_to_k, decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0, decoder_estimator_mid_blocks_9_1_3_attn1_to_out_1, decoder_estimator_mid_blocks_9_1_3_attn1_to_q, decoder_estimator_mid_blocks_9_1_3_attn1_to_v, decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj, decoder_estimator_mid_blocks_9_1_3_ff_net_1, decoder_estimator_mid_blocks_9_1_3_ff_net_2, decoder_estimator_mid_blocks_9_1_3_norm1, decoder_estimator_mid_blocks_9_1_3_norm3, decoder_estimator_time_embeddings, decoder_estimator_time_mlp_act, decoder_estimator_time_mlp_linear_1, decoder_estimator_time_mlp_linear_2, decoder_estimator_up_blocks_0_0_block1_block_0, decoder_estimator_up_blocks_0_0_block1_block_1, decoder_estimator_up_blocks_0_0_block1_block_2, decoder_estimator_up_blocks_0_0_block1_block_3, decoder_estimator_up_blocks_0_0_block1_block_4, decoder_estimator_up_blocks_0_0_block2_block_0, decoder_estimator_up_blocks_0_0_block2_block_1, decoder_estimator_up_blocks_0_0_block2_block_2, decoder_estimator_up_blocks_0_0_block2_block_3, decoder_estimator_up_blocks_0_0_block2_block_4, decoder_estimator_up_blocks_0_0_mlp_0, decoder_estimator_up_blocks_0_0_mlp_1, decoder_estimator_up_blocks_0_0_res_conv, decoder_estimator_up_blocks_0_1_0_attn1_to_k, decoder_estimator_up_blocks_0_1_0_attn1_to_out_0, decoder_estimator_up_blocks_0_1_0_attn1_to_out_1, decoder_estimator_up_blocks_0_1_0_attn1_to_q, decoder_estimator_up_blocks_0_1_0_attn1_to_v, decoder_estimator_up_blocks_0_1_0_ff_net_0_proj, decoder_estimator_up_blocks_0_1_0_ff_net_1, decoder_estimator_up_blocks_0_1_0_ff_net_2, decoder_estimator_up_blocks_0_1_0_norm1, decoder_estimator_up_blocks_0_1_0_norm3, decoder_estimator_up_blocks_0_1_1_attn1_to_k, decoder_estimator_up_blocks_0_1_1_attn1_to_out_0, decoder_estimator_up_blocks_0_1_1_attn1_to_out_1, decoder_estimator_up_blocks_0_1_1_attn1_to_q, decoder_estimator_up_blocks_0_1_1_attn1_to_v, decoder_estimator_up_blocks_0_1_1_ff_net_0_proj, decoder_estimator_up_blocks_0_1_1_ff_net_1, decoder_estimator_up_blocks_0_1_1_ff_net_2, decoder_estimator_up_blocks_0_1_1_norm1, decoder_estimator_up_blocks_0_1_1_norm3, decoder_estimator_up_blocks_0_1_2_attn1_to_k, decoder_estimator_up_blocks_0_1_2_attn1_to_out_0, decoder_estimator_up_blocks_0_1_2_attn1_to_out_1, decoder_estimator_up_blocks_0_1_2_attn1_to_q, decoder_estimator_up_blocks_0_1_2_attn1_to_v, decoder_estimator_up_blocks_0_1_2_ff_net_0_proj, decoder_estimator_up_blocks_0_1_2_ff_net_1, decoder_estimator_up_blocks_0_1_2_ff_net_2, decoder_estimator_up_blocks_0_1_2_norm1, decoder_estimator_up_blocks_0_1_2_norm3, decoder_estimator_up_blocks_0_1_3_attn1_to_k, decoder_estimator_up_blocks_0_1_3_attn1_to_out_0, decoder_estimator_up_blocks_0_1_3_attn1_to_out_1, decoder_estimator_up_blocks_0_1_3_attn1_to_q, decoder_estimator_up_blocks_0_1_3_attn1_to_v, decoder_estimator_up_blocks_0_1_3_ff_net_0_proj, decoder_estimator_up_blocks_0_1_3_ff_net_1, decoder_estimator_up_blocks_0_1_3_ff_net_2, decoder_estimator_up_blocks_0_1_3_norm1, decoder_estimator_up_blocks_0_1_3_norm3, decoder_estimator_up_blocks_0_2, encoder_after_norm, encoder_embed_out_0, encoder_embed_out_1, encoder_embed_out_2, encoder_embed_pos_enc_dropout, encoder_embed_pos_enc_dropout_1, encoder_encoders_0_dropout, encoder_encoders_0_dropout_1, encoder_encoders_0_feed_forward_activation, encoder_encoders_0_feed_forward_activation_1, encoder_encoders_0_feed_forward_activation_2, encoder_encoders_0_feed_forward_activation_3, encoder_encoders_0_feed_forward_activation_4, encoder_encoders_0_feed_forward_activation_5, encoder_encoders_0_feed_forward_activation_6, encoder_encoders_0_feed_forward_activation_7, encoder_encoders_0_feed_forward_activation_8, encoder_encoders_0_feed_forward_activation_9, encoder_encoders_0_feed_forward_dropout, encoder_encoders_0_feed_forward_w_1, encoder_encoders_0_feed_forward_w_2, encoder_encoders_0_norm_ff, encoder_encoders_0_norm_mha, encoder_encoders_0_self_attn_dropout, encoder_encoders_0_self_attn_linear_k, encoder_encoders_0_self_attn_linear_out, encoder_encoders_0_self_attn_linear_pos, encoder_encoders_0_self_attn_linear_q, encoder_encoders_0_self_attn_linear_v, encoder_encoders_1_dropout, encoder_encoders_1_dropout_1, encoder_encoders_1_feed_forward_dropout, encoder_encoders_1_feed_forward_w_1, encoder_encoders_1_feed_forward_w_2, encoder_encoders_1_norm_ff, encoder_encoders_1_norm_mha, encoder_encoders_1_self_attn_dropout, encoder_encoders_1_self_attn_linear_k, encoder_encoders_1_self_attn_linear_out, encoder_encoders_1_self_attn_linear_pos, encoder_encoders_1_self_attn_linear_q, encoder_encoders_1_self_attn_linear_v, encoder_encoders_2_dropout, encoder_encoders_2_dropout_1, encoder_encoders_2_feed_forward_dropout, encoder_encoders_2_feed_forward_w_1, encoder_encoders_2_feed_forward_w_2, encoder_encoders_2_norm_ff, encoder_encoders_2_norm_mha, encoder_encoders_2_self_attn_dropout, encoder_encoders_2_self_attn_linear_k, encoder_encoders_2_self_attn_linear_out, encoder_encoders_2_self_attn_linear_pos, encoder_encoders_2_self_attn_linear_q, encoder_encoders_2_self_attn_linear_v, encoder_encoders_3_dropout, encoder_encoders_3_dropout_1, encoder_encoders_3_feed_forward_dropout, encoder_encoders_3_feed_forward_w_1, encoder_encoders_3_feed_forward_w_2, encoder_encoders_3_norm_ff, encoder_encoders_3_norm_mha, encoder_encoders_3_self_attn_dropout, encoder_encoders_3_self_attn_linear_k, encoder_encoders_3_self_attn_linear_out, encoder_encoders_3_self_attn_linear_pos, encoder_encoders_3_self_attn_linear_q, encoder_encoders_3_self_attn_linear_v, encoder_encoders_4_dropout, encoder_encoders_4_dropout_1, encoder_encoders_4_feed_forward_dropout, encoder_encoders_4_feed_forward_w_1, encoder_encoders_4_feed_forward_w_2, encoder_encoders_4_norm_ff, encoder_encoders_4_norm_mha, encoder_encoders_4_self_attn_dropout, encoder_encoders_4_self_attn_linear_k, encoder_encoders_4_self_attn_linear_out, encoder_encoders_4_self_attn_linear_pos, encoder_encoders_4_self_attn_linear_q, encoder_encoders_4_self_attn_linear_v, encoder_encoders_5_dropout, encoder_encoders_5_dropout_1, encoder_encoders_5_feed_forward_dropout, encoder_encoders_5_feed_forward_w_1, encoder_encoders_5_feed_forward_w_2, encoder_encoders_5_norm_ff, encoder_encoders_5_norm_mha, encoder_encoders_5_self_attn_dropout, encoder_encoders_5_self_attn_linear_k, encoder_encoders_5_self_attn_linear_out, encoder_encoders_5_self_attn_linear_pos, encoder_encoders_5_self_attn_linear_q, encoder_encoders_5_self_attn_linear_v, encoder_pre_lookahead_layer_conv1, encoder_pre_lookahead_layer_conv2, encoder_up_embed_out_0, encoder_up_embed_out_1, encoder_up_embed_out_2, encoder_up_embed_pos_enc_dropout, encoder_up_embed_pos_enc_dropout_1, encoder_up_encoders_0_dropout, encoder_up_encoders_0_dropout_1, encoder_up_encoders_0_feed_forward_dropout, encoder_up_encoders_0_feed_forward_w_1, encoder_up_encoders_0_feed_forward_w_2, encoder_up_encoders_0_norm_ff, encoder_up_encoders_0_norm_mha, encoder_up_encoders_0_self_attn_dropout, encoder_up_encoders_0_self_attn_linear_k, encoder_up_encoders_0_self_attn_linear_out, encoder_up_encoders_0_self_attn_linear_pos, encoder_up_encoders_0_self_attn_linear_q, encoder_up_encoders_0_self_attn_linear_v, encoder_up_encoders_1_dropout, encoder_up_encoders_1_dropout_1, encoder_up_encoders_1_feed_forward_dropout, encoder_up_encoders_1_feed_forward_w_1, encoder_up_encoders_1_feed_forward_w_2, encoder_up_encoders_1_norm_ff, encoder_up_encoders_1_norm_mha, encoder_up_encoders_1_self_attn_dropout, encoder_up_encoders_1_self_attn_linear_k, encoder_up_encoders_1_self_attn_linear_out, encoder_up_encoders_1_self_attn_linear_pos, encoder_up_encoders_1_self_attn_linear_q, encoder_up_encoders_1_self_attn_linear_v, encoder_up_encoders_2_dropout, encoder_up_encoders_2_dropout_1, encoder_up_encoders_2_feed_forward_dropout, encoder_up_encoders_2_feed_forward_w_1, encoder_up_encoders_2_feed_forward_w_2, encoder_up_encoders_2_norm_ff, encoder_up_encoders_2_norm_mha, encoder_up_encoders_2_self_attn_dropout, encoder_up_encoders_2_self_attn_linear_k, encoder_up_encoders_2_self_attn_linear_out, encoder_up_encoders_2_self_attn_linear_pos, encoder_up_encoders_2_self_attn_linear_q, encoder_up_encoders_2_self_attn_linear_v, encoder_up_encoders_3_dropout, encoder_up_encoders_3_dropout_1, encoder_up_encoders_3_feed_forward_dropout, encoder_up_encoders_3_feed_forward_w_1, encoder_up_encoders_3_feed_forward_w_2, encoder_up_encoders_3_norm_ff, encoder_up_encoders_3_norm_mha, encoder_up_encoders_3_self_attn_dropout, encoder_up_encoders_3_self_attn_linear_k, encoder_up_encoders_3_self_attn_linear_out, encoder_up_encoders_3_self_attn_linear_pos, encoder_up_encoders_3_self_attn_linear_q, encoder_up_encoders_3_self_attn_linear_v, encoder_up_layer_conv, encoder_proj, input_embedding, spk_embed_affine_layer, checker })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();

        // Layer: decoder.estimator.down_blocks.0.0.block1.block.0
        x = self.decoder_estimator_down_blocks_0_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block1.block.0", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block1.block.1
        x = self.decoder_estimator_down_blocks_0_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block1.block.1", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block1.block.2
        x = self.decoder_estimator_down_blocks_0_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block1.block.2", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block1.block.3
        x = self.decoder_estimator_down_blocks_0_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block1.block.3", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block1.block.4
        x = self.decoder_estimator_down_blocks_0_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block1.block.4", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block2.block.0
        x = self.decoder_estimator_down_blocks_0_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block2.block.0", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block2.block.1
        x = self.decoder_estimator_down_blocks_0_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block2.block.1", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block2.block.2
        x = self.decoder_estimator_down_blocks_0_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block2.block.2", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block2.block.3
        x = self.decoder_estimator_down_blocks_0_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block2.block.3", &x);

        // Layer: decoder.estimator.down_blocks.0.0.block2.block.4
        x = self.decoder_estimator_down_blocks_0_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.block2.block.4", &x);

        // Layer: decoder.estimator.down_blocks.0.0.mlp.0
        x = self.decoder_estimator_down_blocks_0_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.mlp.0", &x);

        // Layer: decoder.estimator.down_blocks.0.0.mlp.1
        x = self.decoder_estimator_down_blocks_0_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.mlp.1", &x);

        // Layer: decoder.estimator.down_blocks.0.0.res_conv
        x = self.decoder_estimator_down_blocks_0_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.0.res_conv", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.attn1.to_k
        x = self.decoder_estimator_down_blocks_0_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.attn1.to_out.0
        x = self.decoder_estimator_down_blocks_0_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.attn1.to_out.1
        x = self.decoder_estimator_down_blocks_0_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.attn1.to_q
        x = self.decoder_estimator_down_blocks_0_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.attn1.to_v
        x = self.decoder_estimator_down_blocks_0_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.ff.net.0.proj
        x = self.decoder_estimator_down_blocks_0_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.ff.net.1
        x = self.decoder_estimator_down_blocks_0_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.ff.net.2
        x = self.decoder_estimator_down_blocks_0_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.norm1
        x = self.decoder_estimator_down_blocks_0_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.norm1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.0.norm3
        x = self.decoder_estimator_down_blocks_0_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.0.norm3", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.attn1.to_k
        x = self.decoder_estimator_down_blocks_0_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.attn1.to_out.0
        x = self.decoder_estimator_down_blocks_0_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.attn1.to_out.1
        x = self.decoder_estimator_down_blocks_0_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.attn1.to_q
        x = self.decoder_estimator_down_blocks_0_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.attn1.to_v
        x = self.decoder_estimator_down_blocks_0_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.ff.net.0.proj
        x = self.decoder_estimator_down_blocks_0_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.ff.net.1
        x = self.decoder_estimator_down_blocks_0_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.ff.net.2
        x = self.decoder_estimator_down_blocks_0_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.norm1
        x = self.decoder_estimator_down_blocks_0_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.norm1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.1.norm3
        x = self.decoder_estimator_down_blocks_0_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.1.norm3", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.attn1.to_k
        x = self.decoder_estimator_down_blocks_0_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.attn1.to_out.0
        x = self.decoder_estimator_down_blocks_0_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.attn1.to_out.1
        x = self.decoder_estimator_down_blocks_0_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.attn1.to_q
        x = self.decoder_estimator_down_blocks_0_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.attn1.to_v
        x = self.decoder_estimator_down_blocks_0_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.ff.net.0.proj
        x = self.decoder_estimator_down_blocks_0_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.ff.net.1
        x = self.decoder_estimator_down_blocks_0_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.ff.net.2
        x = self.decoder_estimator_down_blocks_0_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.norm1
        x = self.decoder_estimator_down_blocks_0_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.norm1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.2.norm3
        x = self.decoder_estimator_down_blocks_0_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.2.norm3", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.attn1.to_k
        x = self.decoder_estimator_down_blocks_0_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.attn1.to_out.0
        x = self.decoder_estimator_down_blocks_0_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.attn1.to_out.1
        x = self.decoder_estimator_down_blocks_0_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.attn1.to_q
        x = self.decoder_estimator_down_blocks_0_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.attn1.to_v
        x = self.decoder_estimator_down_blocks_0_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.ff.net.0.proj
        x = self.decoder_estimator_down_blocks_0_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.ff.net.1
        x = self.decoder_estimator_down_blocks_0_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.ff.net.2
        x = self.decoder_estimator_down_blocks_0_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.norm1
        x = self.decoder_estimator_down_blocks_0_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.norm1", &x);

        // Layer: decoder.estimator.down_blocks.0.1.3.norm3
        x = self.decoder_estimator_down_blocks_0_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.1.3.norm3", &x);

        // Layer: decoder.estimator.down_blocks.0.2
        x = self.decoder_estimator_down_blocks_0_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.down_blocks.0.2", &x);

        // Layer: decoder.estimator.final_block.block.0
        x = self.decoder_estimator_final_block_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_block.block.0", &x);

        // Layer: decoder.estimator.final_block.block.1
        x = self.decoder_estimator_final_block_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_block.block.1", &x);

        // Layer: decoder.estimator.final_block.block.2
        x = self.decoder_estimator_final_block_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_block.block.2", &x);

        // Layer: decoder.estimator.final_block.block.3
        x = self.decoder_estimator_final_block_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_block.block.3", &x);

        // Layer: decoder.estimator.final_block.block.4
        x = self.decoder_estimator_final_block_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_block.block.4", &x);

        // Layer: decoder.estimator.final_proj
        x = self.decoder_estimator_final_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.final_proj", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_0_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_0_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_0_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_0_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_0_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_0_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_0_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_0_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_0_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_0_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.mlp.0
        x = self.decoder_estimator_mid_blocks_0_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.mlp.1
        x = self.decoder_estimator_mid_blocks_0_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.0.res_conv
        x = self.decoder_estimator_mid_blocks_0_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_0_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_0_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_0_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_0_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_0_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_0_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_0_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_0_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.norm1
        x = self.decoder_estimator_mid_blocks_0_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.0.norm3
        x = self.decoder_estimator_mid_blocks_0_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_0_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_0_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_0_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_0_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_0_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_0_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_0_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_0_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.norm1
        x = self.decoder_estimator_mid_blocks_0_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.1.norm3
        x = self.decoder_estimator_mid_blocks_0_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_0_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_0_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_0_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_0_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_0_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_0_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_0_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_0_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.norm1
        x = self.decoder_estimator_mid_blocks_0_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.2.norm3
        x = self.decoder_estimator_mid_blocks_0_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_0_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_0_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_0_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_0_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_0_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_0_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_0_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_0_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.norm1
        x = self.decoder_estimator_mid_blocks_0_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.0.1.3.norm3
        x = self.decoder_estimator_mid_blocks_0_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.0.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_1_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_1_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_1_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_1_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_1_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_1_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_1_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_1_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_1_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_1_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.mlp.0
        x = self.decoder_estimator_mid_blocks_1_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.mlp.1
        x = self.decoder_estimator_mid_blocks_1_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.0.res_conv
        x = self.decoder_estimator_mid_blocks_1_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_1_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_1_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_1_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_1_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_1_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_1_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_1_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_1_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.norm1
        x = self.decoder_estimator_mid_blocks_1_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.0.norm3
        x = self.decoder_estimator_mid_blocks_1_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_1_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_1_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_1_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_1_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_1_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_1_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_1_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_1_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.norm1
        x = self.decoder_estimator_mid_blocks_1_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.1.norm3
        x = self.decoder_estimator_mid_blocks_1_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_1_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_1_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_1_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_1_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_1_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_1_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_1_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_1_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.norm1
        x = self.decoder_estimator_mid_blocks_1_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.2.norm3
        x = self.decoder_estimator_mid_blocks_1_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_1_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_1_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_1_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_1_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_1_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_1_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_1_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_1_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.norm1
        x = self.decoder_estimator_mid_blocks_1_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.1.1.3.norm3
        x = self.decoder_estimator_mid_blocks_1_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.1.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_10_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_10_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_10_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_10_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_10_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_10_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_10_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_10_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_10_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_10_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.mlp.0
        x = self.decoder_estimator_mid_blocks_10_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.mlp.1
        x = self.decoder_estimator_mid_blocks_10_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.0.res_conv
        x = self.decoder_estimator_mid_blocks_10_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_10_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_10_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_10_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_10_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_10_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_10_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_10_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_10_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.norm1
        x = self.decoder_estimator_mid_blocks_10_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.0.norm3
        x = self.decoder_estimator_mid_blocks_10_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_10_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_10_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_10_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_10_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_10_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_10_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_10_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_10_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.norm1
        x = self.decoder_estimator_mid_blocks_10_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.1.norm3
        x = self.decoder_estimator_mid_blocks_10_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_10_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_10_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_10_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_10_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_10_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_10_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_10_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_10_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.norm1
        x = self.decoder_estimator_mid_blocks_10_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.2.norm3
        x = self.decoder_estimator_mid_blocks_10_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_10_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_10_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_10_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_10_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_10_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_10_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_10_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_10_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.norm1
        x = self.decoder_estimator_mid_blocks_10_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.10.1.3.norm3
        x = self.decoder_estimator_mid_blocks_10_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.10.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_11_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_11_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_11_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_11_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_11_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_11_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_11_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_11_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_11_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_11_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.mlp.0
        x = self.decoder_estimator_mid_blocks_11_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.mlp.1
        x = self.decoder_estimator_mid_blocks_11_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.0.res_conv
        x = self.decoder_estimator_mid_blocks_11_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_11_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_11_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_11_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_11_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_11_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_11_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_11_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_11_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.norm1
        x = self.decoder_estimator_mid_blocks_11_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.0.norm3
        x = self.decoder_estimator_mid_blocks_11_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_11_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_11_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_11_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_11_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_11_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_11_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_11_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_11_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.norm1
        x = self.decoder_estimator_mid_blocks_11_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.1.norm3
        x = self.decoder_estimator_mid_blocks_11_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_11_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_11_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_11_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_11_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_11_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_11_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_11_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_11_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.norm1
        x = self.decoder_estimator_mid_blocks_11_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.2.norm3
        x = self.decoder_estimator_mid_blocks_11_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_11_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_11_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_11_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_11_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_11_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_11_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_11_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_11_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.norm1
        x = self.decoder_estimator_mid_blocks_11_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.11.1.3.norm3
        x = self.decoder_estimator_mid_blocks_11_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.11.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_2_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_2_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_2_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_2_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_2_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_2_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_2_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_2_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_2_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_2_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.mlp.0
        x = self.decoder_estimator_mid_blocks_2_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.mlp.1
        x = self.decoder_estimator_mid_blocks_2_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.0.res_conv
        x = self.decoder_estimator_mid_blocks_2_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_2_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_2_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_2_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_2_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_2_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_2_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_2_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_2_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.norm1
        x = self.decoder_estimator_mid_blocks_2_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.0.norm3
        x = self.decoder_estimator_mid_blocks_2_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_2_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_2_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_2_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_2_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_2_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_2_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_2_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_2_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.norm1
        x = self.decoder_estimator_mid_blocks_2_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.1.norm3
        x = self.decoder_estimator_mid_blocks_2_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_2_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_2_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_2_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_2_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_2_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_2_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_2_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_2_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.norm1
        x = self.decoder_estimator_mid_blocks_2_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.2.norm3
        x = self.decoder_estimator_mid_blocks_2_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_2_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_2_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_2_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_2_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_2_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_2_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_2_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_2_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.norm1
        x = self.decoder_estimator_mid_blocks_2_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.2.1.3.norm3
        x = self.decoder_estimator_mid_blocks_2_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.2.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_3_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_3_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_3_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_3_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_3_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_3_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_3_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_3_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_3_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_3_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.mlp.0
        x = self.decoder_estimator_mid_blocks_3_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.mlp.1
        x = self.decoder_estimator_mid_blocks_3_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.0.res_conv
        x = self.decoder_estimator_mid_blocks_3_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_3_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_3_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_3_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_3_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_3_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_3_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_3_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_3_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.norm1
        x = self.decoder_estimator_mid_blocks_3_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.0.norm3
        x = self.decoder_estimator_mid_blocks_3_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_3_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_3_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_3_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_3_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_3_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_3_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_3_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_3_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.norm1
        x = self.decoder_estimator_mid_blocks_3_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.1.norm3
        x = self.decoder_estimator_mid_blocks_3_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_3_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_3_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_3_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_3_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_3_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_3_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_3_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_3_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.norm1
        x = self.decoder_estimator_mid_blocks_3_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.2.norm3
        x = self.decoder_estimator_mid_blocks_3_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_3_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_3_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_3_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_3_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_3_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_3_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_3_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_3_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.norm1
        x = self.decoder_estimator_mid_blocks_3_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.3.1.3.norm3
        x = self.decoder_estimator_mid_blocks_3_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.3.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_4_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_4_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_4_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_4_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_4_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_4_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_4_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_4_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_4_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_4_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.mlp.0
        x = self.decoder_estimator_mid_blocks_4_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.mlp.1
        x = self.decoder_estimator_mid_blocks_4_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.0.res_conv
        x = self.decoder_estimator_mid_blocks_4_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_4_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_4_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_4_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_4_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_4_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_4_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_4_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_4_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.norm1
        x = self.decoder_estimator_mid_blocks_4_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.0.norm3
        x = self.decoder_estimator_mid_blocks_4_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_4_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_4_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_4_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_4_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_4_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_4_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_4_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_4_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.norm1
        x = self.decoder_estimator_mid_blocks_4_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.1.norm3
        x = self.decoder_estimator_mid_blocks_4_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_4_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_4_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_4_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_4_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_4_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_4_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_4_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_4_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.norm1
        x = self.decoder_estimator_mid_blocks_4_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.2.norm3
        x = self.decoder_estimator_mid_blocks_4_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_4_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_4_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_4_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_4_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_4_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_4_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_4_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_4_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.norm1
        x = self.decoder_estimator_mid_blocks_4_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.4.1.3.norm3
        x = self.decoder_estimator_mid_blocks_4_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.4.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_5_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_5_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_5_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_5_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_5_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_5_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_5_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_5_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_5_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_5_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.mlp.0
        x = self.decoder_estimator_mid_blocks_5_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.mlp.1
        x = self.decoder_estimator_mid_blocks_5_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.0.res_conv
        x = self.decoder_estimator_mid_blocks_5_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_5_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_5_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_5_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_5_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_5_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_5_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_5_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_5_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.norm1
        x = self.decoder_estimator_mid_blocks_5_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.0.norm3
        x = self.decoder_estimator_mid_blocks_5_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_5_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_5_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_5_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_5_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_5_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_5_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_5_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_5_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.norm1
        x = self.decoder_estimator_mid_blocks_5_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.1.norm3
        x = self.decoder_estimator_mid_blocks_5_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_5_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_5_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_5_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_5_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_5_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_5_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_5_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_5_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.norm1
        x = self.decoder_estimator_mid_blocks_5_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.2.norm3
        x = self.decoder_estimator_mid_blocks_5_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_5_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_5_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_5_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_5_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_5_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_5_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_5_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_5_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.norm1
        x = self.decoder_estimator_mid_blocks_5_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.5.1.3.norm3
        x = self.decoder_estimator_mid_blocks_5_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.5.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_6_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_6_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_6_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_6_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_6_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_6_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_6_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_6_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_6_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_6_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.mlp.0
        x = self.decoder_estimator_mid_blocks_6_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.mlp.1
        x = self.decoder_estimator_mid_blocks_6_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.0.res_conv
        x = self.decoder_estimator_mid_blocks_6_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_6_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_6_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_6_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_6_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_6_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_6_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_6_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_6_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.norm1
        x = self.decoder_estimator_mid_blocks_6_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.0.norm3
        x = self.decoder_estimator_mid_blocks_6_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_6_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_6_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_6_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_6_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_6_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_6_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_6_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_6_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.norm1
        x = self.decoder_estimator_mid_blocks_6_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.1.norm3
        x = self.decoder_estimator_mid_blocks_6_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_6_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_6_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_6_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_6_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_6_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_6_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_6_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_6_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.norm1
        x = self.decoder_estimator_mid_blocks_6_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.2.norm3
        x = self.decoder_estimator_mid_blocks_6_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_6_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_6_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_6_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_6_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_6_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_6_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_6_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_6_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.norm1
        x = self.decoder_estimator_mid_blocks_6_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.6.1.3.norm3
        x = self.decoder_estimator_mid_blocks_6_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.6.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_7_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_7_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_7_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_7_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_7_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_7_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_7_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_7_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_7_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_7_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.mlp.0
        x = self.decoder_estimator_mid_blocks_7_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.mlp.1
        x = self.decoder_estimator_mid_blocks_7_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.0.res_conv
        x = self.decoder_estimator_mid_blocks_7_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_7_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_7_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_7_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_7_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_7_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_7_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_7_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_7_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.norm1
        x = self.decoder_estimator_mid_blocks_7_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.0.norm3
        x = self.decoder_estimator_mid_blocks_7_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_7_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_7_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_7_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_7_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_7_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_7_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_7_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_7_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.norm1
        x = self.decoder_estimator_mid_blocks_7_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.1.norm3
        x = self.decoder_estimator_mid_blocks_7_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_7_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_7_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_7_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_7_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_7_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_7_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_7_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_7_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.norm1
        x = self.decoder_estimator_mid_blocks_7_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.2.norm3
        x = self.decoder_estimator_mid_blocks_7_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_7_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_7_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_7_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_7_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_7_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_7_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_7_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_7_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.norm1
        x = self.decoder_estimator_mid_blocks_7_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.7.1.3.norm3
        x = self.decoder_estimator_mid_blocks_7_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.7.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_8_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_8_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_8_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_8_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_8_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_8_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_8_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_8_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_8_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_8_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.mlp.0
        x = self.decoder_estimator_mid_blocks_8_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.mlp.1
        x = self.decoder_estimator_mid_blocks_8_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.0.res_conv
        x = self.decoder_estimator_mid_blocks_8_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_8_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_8_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_8_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_8_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_8_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_8_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_8_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_8_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.norm1
        x = self.decoder_estimator_mid_blocks_8_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.0.norm3
        x = self.decoder_estimator_mid_blocks_8_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_8_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_8_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_8_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_8_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_8_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_8_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_8_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_8_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.norm1
        x = self.decoder_estimator_mid_blocks_8_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.1.norm3
        x = self.decoder_estimator_mid_blocks_8_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_8_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_8_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_8_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_8_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_8_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_8_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_8_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_8_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.norm1
        x = self.decoder_estimator_mid_blocks_8_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.2.norm3
        x = self.decoder_estimator_mid_blocks_8_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_8_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_8_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_8_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_8_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_8_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_8_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_8_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_8_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.norm1
        x = self.decoder_estimator_mid_blocks_8_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.8.1.3.norm3
        x = self.decoder_estimator_mid_blocks_8_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.8.1.3.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block1.block.0
        x = self.decoder_estimator_mid_blocks_9_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block1.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block1.block.1
        x = self.decoder_estimator_mid_blocks_9_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block1.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block1.block.2
        x = self.decoder_estimator_mid_blocks_9_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block1.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block1.block.3
        x = self.decoder_estimator_mid_blocks_9_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block1.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block1.block.4
        x = self.decoder_estimator_mid_blocks_9_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block1.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block2.block.0
        x = self.decoder_estimator_mid_blocks_9_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block2.block.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block2.block.1
        x = self.decoder_estimator_mid_blocks_9_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block2.block.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block2.block.2
        x = self.decoder_estimator_mid_blocks_9_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block2.block.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block2.block.3
        x = self.decoder_estimator_mid_blocks_9_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block2.block.3", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.block2.block.4
        x = self.decoder_estimator_mid_blocks_9_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.block2.block.4", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.mlp.0
        x = self.decoder_estimator_mid_blocks_9_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.mlp.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.mlp.1
        x = self.decoder_estimator_mid_blocks_9_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.mlp.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.0.res_conv
        x = self.decoder_estimator_mid_blocks_9_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.0.res_conv", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.attn1.to_k
        x = self.decoder_estimator_mid_blocks_9_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_9_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_9_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.attn1.to_q
        x = self.decoder_estimator_mid_blocks_9_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.attn1.to_v
        x = self.decoder_estimator_mid_blocks_9_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_9_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.ff.net.1
        x = self.decoder_estimator_mid_blocks_9_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.ff.net.2
        x = self.decoder_estimator_mid_blocks_9_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.norm1
        x = self.decoder_estimator_mid_blocks_9_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.0.norm3
        x = self.decoder_estimator_mid_blocks_9_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.0.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.attn1.to_k
        x = self.decoder_estimator_mid_blocks_9_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_9_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_9_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.attn1.to_q
        x = self.decoder_estimator_mid_blocks_9_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.attn1.to_v
        x = self.decoder_estimator_mid_blocks_9_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_9_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.ff.net.1
        x = self.decoder_estimator_mid_blocks_9_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.ff.net.2
        x = self.decoder_estimator_mid_blocks_9_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.norm1
        x = self.decoder_estimator_mid_blocks_9_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.1.norm3
        x = self.decoder_estimator_mid_blocks_9_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.1.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.attn1.to_k
        x = self.decoder_estimator_mid_blocks_9_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_9_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_9_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.attn1.to_q
        x = self.decoder_estimator_mid_blocks_9_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.attn1.to_v
        x = self.decoder_estimator_mid_blocks_9_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_9_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.ff.net.1
        x = self.decoder_estimator_mid_blocks_9_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.ff.net.2
        x = self.decoder_estimator_mid_blocks_9_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.norm1
        x = self.decoder_estimator_mid_blocks_9_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.2.norm3
        x = self.decoder_estimator_mid_blocks_9_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.2.norm3", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.attn1.to_k
        x = self.decoder_estimator_mid_blocks_9_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.attn1.to_out.0
        x = self.decoder_estimator_mid_blocks_9_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.attn1.to_out.1
        x = self.decoder_estimator_mid_blocks_9_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.attn1.to_q
        x = self.decoder_estimator_mid_blocks_9_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.attn1.to_v
        x = self.decoder_estimator_mid_blocks_9_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.ff.net.0.proj
        x = self.decoder_estimator_mid_blocks_9_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.ff.net.1
        x = self.decoder_estimator_mid_blocks_9_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.ff.net.2
        x = self.decoder_estimator_mid_blocks_9_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.norm1
        x = self.decoder_estimator_mid_blocks_9_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.norm1", &x);

        // Layer: decoder.estimator.mid_blocks.9.1.3.norm3
        x = self.decoder_estimator_mid_blocks_9_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.mid_blocks.9.1.3.norm3", &x);

        // Layer: decoder.estimator.time_embeddings
        x = self.decoder_estimator_time_embeddings.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.time_embeddings", &x);

        // Layer: decoder.estimator.time_mlp.act
        x = self.decoder_estimator_time_mlp_act.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.time_mlp.act", &x);

        // Layer: decoder.estimator.time_mlp.linear_1
        x = self.decoder_estimator_time_mlp_linear_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.time_mlp.linear_1", &x);

        // Layer: decoder.estimator.time_mlp.linear_2
        x = self.decoder_estimator_time_mlp_linear_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.time_mlp.linear_2", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block1.block.0
        x = self.decoder_estimator_up_blocks_0_0_block1_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block1.block.0", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block1.block.1
        x = self.decoder_estimator_up_blocks_0_0_block1_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block1.block.1", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block1.block.2
        x = self.decoder_estimator_up_blocks_0_0_block1_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block1.block.2", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block1.block.3
        x = self.decoder_estimator_up_blocks_0_0_block1_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block1.block.3", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block1.block.4
        x = self.decoder_estimator_up_blocks_0_0_block1_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block1.block.4", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block2.block.0
        x = self.decoder_estimator_up_blocks_0_0_block2_block_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block2.block.0", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block2.block.1
        x = self.decoder_estimator_up_blocks_0_0_block2_block_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block2.block.1", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block2.block.2
        x = self.decoder_estimator_up_blocks_0_0_block2_block_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block2.block.2", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block2.block.3
        x = self.decoder_estimator_up_blocks_0_0_block2_block_3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block2.block.3", &x);

        // Layer: decoder.estimator.up_blocks.0.0.block2.block.4
        x = self.decoder_estimator_up_blocks_0_0_block2_block_4.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.block2.block.4", &x);

        // Layer: decoder.estimator.up_blocks.0.0.mlp.0
        x = self.decoder_estimator_up_blocks_0_0_mlp_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.mlp.0", &x);

        // Layer: decoder.estimator.up_blocks.0.0.mlp.1
        x = self.decoder_estimator_up_blocks_0_0_mlp_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.mlp.1", &x);

        // Layer: decoder.estimator.up_blocks.0.0.res_conv
        x = self.decoder_estimator_up_blocks_0_0_res_conv.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.0.res_conv", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.attn1.to_k
        x = self.decoder_estimator_up_blocks_0_1_0_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.attn1.to_k", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.attn1.to_out.0
        x = self.decoder_estimator_up_blocks_0_1_0_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.attn1.to_out.0", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.attn1.to_out.1
        x = self.decoder_estimator_up_blocks_0_1_0_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.attn1.to_out.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.attn1.to_q
        x = self.decoder_estimator_up_blocks_0_1_0_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.attn1.to_q", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.attn1.to_v
        x = self.decoder_estimator_up_blocks_0_1_0_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.attn1.to_v", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.ff.net.0.proj
        x = self.decoder_estimator_up_blocks_0_1_0_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.ff.net.0.proj", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.ff.net.1
        x = self.decoder_estimator_up_blocks_0_1_0_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.ff.net.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.ff.net.2
        x = self.decoder_estimator_up_blocks_0_1_0_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.ff.net.2", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.norm1
        x = self.decoder_estimator_up_blocks_0_1_0_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.norm1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.0.norm3
        x = self.decoder_estimator_up_blocks_0_1_0_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.0.norm3", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.attn1.to_k
        x = self.decoder_estimator_up_blocks_0_1_1_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.attn1.to_k", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.attn1.to_out.0
        x = self.decoder_estimator_up_blocks_0_1_1_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.attn1.to_out.0", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.attn1.to_out.1
        x = self.decoder_estimator_up_blocks_0_1_1_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.attn1.to_out.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.attn1.to_q
        x = self.decoder_estimator_up_blocks_0_1_1_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.attn1.to_q", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.attn1.to_v
        x = self.decoder_estimator_up_blocks_0_1_1_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.attn1.to_v", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.ff.net.0.proj
        x = self.decoder_estimator_up_blocks_0_1_1_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.ff.net.0.proj", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.ff.net.1
        x = self.decoder_estimator_up_blocks_0_1_1_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.ff.net.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.ff.net.2
        x = self.decoder_estimator_up_blocks_0_1_1_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.ff.net.2", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.norm1
        x = self.decoder_estimator_up_blocks_0_1_1_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.norm1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.1.norm3
        x = self.decoder_estimator_up_blocks_0_1_1_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.1.norm3", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.attn1.to_k
        x = self.decoder_estimator_up_blocks_0_1_2_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.attn1.to_k", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.attn1.to_out.0
        x = self.decoder_estimator_up_blocks_0_1_2_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.attn1.to_out.0", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.attn1.to_out.1
        x = self.decoder_estimator_up_blocks_0_1_2_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.attn1.to_out.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.attn1.to_q
        x = self.decoder_estimator_up_blocks_0_1_2_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.attn1.to_q", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.attn1.to_v
        x = self.decoder_estimator_up_blocks_0_1_2_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.attn1.to_v", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.ff.net.0.proj
        x = self.decoder_estimator_up_blocks_0_1_2_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.ff.net.0.proj", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.ff.net.1
        x = self.decoder_estimator_up_blocks_0_1_2_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.ff.net.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.ff.net.2
        x = self.decoder_estimator_up_blocks_0_1_2_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.ff.net.2", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.norm1
        x = self.decoder_estimator_up_blocks_0_1_2_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.norm1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.2.norm3
        x = self.decoder_estimator_up_blocks_0_1_2_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.2.norm3", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.attn1.to_k
        x = self.decoder_estimator_up_blocks_0_1_3_attn1_to_k.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.attn1.to_k", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.attn1.to_out.0
        x = self.decoder_estimator_up_blocks_0_1_3_attn1_to_out_0.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.attn1.to_out.0", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.attn1.to_out.1
        x = self.decoder_estimator_up_blocks_0_1_3_attn1_to_out_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.attn1.to_out.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.attn1.to_q
        x = self.decoder_estimator_up_blocks_0_1_3_attn1_to_q.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.attn1.to_q", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.attn1.to_v
        x = self.decoder_estimator_up_blocks_0_1_3_attn1_to_v.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.attn1.to_v", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.ff.net.0.proj
        x = self.decoder_estimator_up_blocks_0_1_3_ff_net_0_proj.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.ff.net.0.proj", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.ff.net.1
        x = self.decoder_estimator_up_blocks_0_1_3_ff_net_1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.ff.net.1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.ff.net.2
        x = self.decoder_estimator_up_blocks_0_1_3_ff_net_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.ff.net.2", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.norm1
        x = self.decoder_estimator_up_blocks_0_1_3_norm1.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.norm1", &x);

        // Layer: decoder.estimator.up_blocks.0.1.3.norm3
        x = self.decoder_estimator_up_blocks_0_1_3_norm3.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.1.3.norm3", &x);

        // Layer: decoder.estimator.up_blocks.0.2
        x = self.decoder_estimator_up_blocks_0_2.forward(&x)?;
        py_check!(self.checker, "decoder.estimator.up_blocks.0.2", &x);

        // Layer: encoder.after_norm
        x = self.encoder_after_norm.forward(&x)?;
        py_check!(self.checker, "encoder.after_norm", &x);

        // Layer: encoder.embed.out.0
        x = self.encoder_embed_out_0.forward(&x)?;
        py_check!(self.checker, "encoder.embed.out.0", &x);

        // Layer: encoder.embed.out.1
        x = self.encoder_embed_out_1.forward(&x)?;
        py_check!(self.checker, "encoder.embed.out.1", &x);

        // Layer: encoder.embed.out.2
        x = self.encoder_embed_out_2.forward(&x)?;
        py_check!(self.checker, "encoder.embed.out.2", &x);

        // Layer: encoder.embed.pos_enc.dropout
        x = self.encoder_embed_pos_enc_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.embed.pos_enc.dropout", &x);

        // Layer: encoder.embed.pos_enc.dropout.1
        x = self.encoder_embed_pos_enc_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.embed.pos_enc.dropout.1", &x);

        // Layer: encoder.encoders.0.dropout
        x = self.encoder_encoders_0_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.dropout", &x);

        // Layer: encoder.encoders.0.dropout.1
        x = self.encoder_encoders_0_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.dropout.1", &x);

        // Layer: encoder.encoders.0.feed_forward.activation
        x = self.encoder_encoders_0_feed_forward_activation.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.1
        x = self.encoder_encoders_0_feed_forward_activation_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.1", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.2
        x = self.encoder_encoders_0_feed_forward_activation_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.2", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.3
        x = self.encoder_encoders_0_feed_forward_activation_3.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.3", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.4
        x = self.encoder_encoders_0_feed_forward_activation_4.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.4", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.5
        x = self.encoder_encoders_0_feed_forward_activation_5.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.5", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.6
        x = self.encoder_encoders_0_feed_forward_activation_6.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.6", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.7
        x = self.encoder_encoders_0_feed_forward_activation_7.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.7", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.8
        x = self.encoder_encoders_0_feed_forward_activation_8.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.8", &x);

        // Layer: encoder.encoders.0.feed_forward.activation.9
        x = self.encoder_encoders_0_feed_forward_activation_9.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.activation.9", &x);

        // Layer: encoder.encoders.0.feed_forward.dropout
        x = self.encoder_encoders_0_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.dropout", &x);

        // Layer: encoder.encoders.0.feed_forward.w_1
        x = self.encoder_encoders_0_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.w_1", &x);

        // Layer: encoder.encoders.0.feed_forward.w_2
        x = self.encoder_encoders_0_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.feed_forward.w_2", &x);

        // Layer: encoder.encoders.0.norm_ff
        x = self.encoder_encoders_0_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.norm_ff", &x);

        // Layer: encoder.encoders.0.norm_mha
        x = self.encoder_encoders_0_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.norm_mha", &x);

        // Layer: encoder.encoders.0.self_attn.dropout
        x = self.encoder_encoders_0_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.dropout", &x);

        // Layer: encoder.encoders.0.self_attn.linear_k
        x = self.encoder_encoders_0_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.linear_k", &x);

        // Layer: encoder.encoders.0.self_attn.linear_out
        x = self.encoder_encoders_0_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.linear_out", &x);

        // Layer: encoder.encoders.0.self_attn.linear_pos
        x = self.encoder_encoders_0_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.0.self_attn.linear_q
        x = self.encoder_encoders_0_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.linear_q", &x);

        // Layer: encoder.encoders.0.self_attn.linear_v
        x = self.encoder_encoders_0_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.0.self_attn.linear_v", &x);

        // Layer: encoder.encoders.1.dropout
        x = self.encoder_encoders_1_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.dropout", &x);

        // Layer: encoder.encoders.1.dropout.1
        x = self.encoder_encoders_1_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.dropout.1", &x);

        // Layer: encoder.encoders.1.feed_forward.dropout
        x = self.encoder_encoders_1_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.feed_forward.dropout", &x);

        // Layer: encoder.encoders.1.feed_forward.w_1
        x = self.encoder_encoders_1_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.feed_forward.w_1", &x);

        // Layer: encoder.encoders.1.feed_forward.w_2
        x = self.encoder_encoders_1_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.feed_forward.w_2", &x);

        // Layer: encoder.encoders.1.norm_ff
        x = self.encoder_encoders_1_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.norm_ff", &x);

        // Layer: encoder.encoders.1.norm_mha
        x = self.encoder_encoders_1_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.norm_mha", &x);

        // Layer: encoder.encoders.1.self_attn.dropout
        x = self.encoder_encoders_1_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.dropout", &x);

        // Layer: encoder.encoders.1.self_attn.linear_k
        x = self.encoder_encoders_1_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.linear_k", &x);

        // Layer: encoder.encoders.1.self_attn.linear_out
        x = self.encoder_encoders_1_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.linear_out", &x);

        // Layer: encoder.encoders.1.self_attn.linear_pos
        x = self.encoder_encoders_1_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.1.self_attn.linear_q
        x = self.encoder_encoders_1_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.linear_q", &x);

        // Layer: encoder.encoders.1.self_attn.linear_v
        x = self.encoder_encoders_1_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.1.self_attn.linear_v", &x);

        // Layer: encoder.encoders.2.dropout
        x = self.encoder_encoders_2_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.dropout", &x);

        // Layer: encoder.encoders.2.dropout.1
        x = self.encoder_encoders_2_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.dropout.1", &x);

        // Layer: encoder.encoders.2.feed_forward.dropout
        x = self.encoder_encoders_2_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.feed_forward.dropout", &x);

        // Layer: encoder.encoders.2.feed_forward.w_1
        x = self.encoder_encoders_2_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.feed_forward.w_1", &x);

        // Layer: encoder.encoders.2.feed_forward.w_2
        x = self.encoder_encoders_2_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.feed_forward.w_2", &x);

        // Layer: encoder.encoders.2.norm_ff
        x = self.encoder_encoders_2_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.norm_ff", &x);

        // Layer: encoder.encoders.2.norm_mha
        x = self.encoder_encoders_2_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.norm_mha", &x);

        // Layer: encoder.encoders.2.self_attn.dropout
        x = self.encoder_encoders_2_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.dropout", &x);

        // Layer: encoder.encoders.2.self_attn.linear_k
        x = self.encoder_encoders_2_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.linear_k", &x);

        // Layer: encoder.encoders.2.self_attn.linear_out
        x = self.encoder_encoders_2_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.linear_out", &x);

        // Layer: encoder.encoders.2.self_attn.linear_pos
        x = self.encoder_encoders_2_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.2.self_attn.linear_q
        x = self.encoder_encoders_2_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.linear_q", &x);

        // Layer: encoder.encoders.2.self_attn.linear_v
        x = self.encoder_encoders_2_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.2.self_attn.linear_v", &x);

        // Layer: encoder.encoders.3.dropout
        x = self.encoder_encoders_3_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.dropout", &x);

        // Layer: encoder.encoders.3.dropout.1
        x = self.encoder_encoders_3_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.dropout.1", &x);

        // Layer: encoder.encoders.3.feed_forward.dropout
        x = self.encoder_encoders_3_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.feed_forward.dropout", &x);

        // Layer: encoder.encoders.3.feed_forward.w_1
        x = self.encoder_encoders_3_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.feed_forward.w_1", &x);

        // Layer: encoder.encoders.3.feed_forward.w_2
        x = self.encoder_encoders_3_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.feed_forward.w_2", &x);

        // Layer: encoder.encoders.3.norm_ff
        x = self.encoder_encoders_3_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.norm_ff", &x);

        // Layer: encoder.encoders.3.norm_mha
        x = self.encoder_encoders_3_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.norm_mha", &x);

        // Layer: encoder.encoders.3.self_attn.dropout
        x = self.encoder_encoders_3_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.dropout", &x);

        // Layer: encoder.encoders.3.self_attn.linear_k
        x = self.encoder_encoders_3_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.linear_k", &x);

        // Layer: encoder.encoders.3.self_attn.linear_out
        x = self.encoder_encoders_3_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.linear_out", &x);

        // Layer: encoder.encoders.3.self_attn.linear_pos
        x = self.encoder_encoders_3_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.3.self_attn.linear_q
        x = self.encoder_encoders_3_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.linear_q", &x);

        // Layer: encoder.encoders.3.self_attn.linear_v
        x = self.encoder_encoders_3_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.3.self_attn.linear_v", &x);

        // Layer: encoder.encoders.4.dropout
        x = self.encoder_encoders_4_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.dropout", &x);

        // Layer: encoder.encoders.4.dropout.1
        x = self.encoder_encoders_4_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.dropout.1", &x);

        // Layer: encoder.encoders.4.feed_forward.dropout
        x = self.encoder_encoders_4_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.feed_forward.dropout", &x);

        // Layer: encoder.encoders.4.feed_forward.w_1
        x = self.encoder_encoders_4_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.feed_forward.w_1", &x);

        // Layer: encoder.encoders.4.feed_forward.w_2
        x = self.encoder_encoders_4_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.feed_forward.w_2", &x);

        // Layer: encoder.encoders.4.norm_ff
        x = self.encoder_encoders_4_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.norm_ff", &x);

        // Layer: encoder.encoders.4.norm_mha
        x = self.encoder_encoders_4_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.norm_mha", &x);

        // Layer: encoder.encoders.4.self_attn.dropout
        x = self.encoder_encoders_4_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.dropout", &x);

        // Layer: encoder.encoders.4.self_attn.linear_k
        x = self.encoder_encoders_4_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.linear_k", &x);

        // Layer: encoder.encoders.4.self_attn.linear_out
        x = self.encoder_encoders_4_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.linear_out", &x);

        // Layer: encoder.encoders.4.self_attn.linear_pos
        x = self.encoder_encoders_4_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.4.self_attn.linear_q
        x = self.encoder_encoders_4_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.linear_q", &x);

        // Layer: encoder.encoders.4.self_attn.linear_v
        x = self.encoder_encoders_4_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.4.self_attn.linear_v", &x);

        // Layer: encoder.encoders.5.dropout
        x = self.encoder_encoders_5_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.dropout", &x);

        // Layer: encoder.encoders.5.dropout.1
        x = self.encoder_encoders_5_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.dropout.1", &x);

        // Layer: encoder.encoders.5.feed_forward.dropout
        x = self.encoder_encoders_5_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.feed_forward.dropout", &x);

        // Layer: encoder.encoders.5.feed_forward.w_1
        x = self.encoder_encoders_5_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.feed_forward.w_1", &x);

        // Layer: encoder.encoders.5.feed_forward.w_2
        x = self.encoder_encoders_5_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.feed_forward.w_2", &x);

        // Layer: encoder.encoders.5.norm_ff
        x = self.encoder_encoders_5_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.norm_ff", &x);

        // Layer: encoder.encoders.5.norm_mha
        x = self.encoder_encoders_5_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.norm_mha", &x);

        // Layer: encoder.encoders.5.self_attn.dropout
        x = self.encoder_encoders_5_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.dropout", &x);

        // Layer: encoder.encoders.5.self_attn.linear_k
        x = self.encoder_encoders_5_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.linear_k", &x);

        // Layer: encoder.encoders.5.self_attn.linear_out
        x = self.encoder_encoders_5_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.linear_out", &x);

        // Layer: encoder.encoders.5.self_attn.linear_pos
        x = self.encoder_encoders_5_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.linear_pos", &x);

        // Layer: encoder.encoders.5.self_attn.linear_q
        x = self.encoder_encoders_5_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.linear_q", &x);

        // Layer: encoder.encoders.5.self_attn.linear_v
        x = self.encoder_encoders_5_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.encoders.5.self_attn.linear_v", &x);

        // Layer: encoder.pre_lookahead_layer.conv1
        x = self.encoder_pre_lookahead_layer_conv1.forward(&x)?;
        py_check!(self.checker, "encoder.pre_lookahead_layer.conv1", &x);

        // Layer: encoder.pre_lookahead_layer.conv2
        x = self.encoder_pre_lookahead_layer_conv2.forward(&x)?;
        py_check!(self.checker, "encoder.pre_lookahead_layer.conv2", &x);

        // Layer: encoder.up_embed.out.0
        x = self.encoder_up_embed_out_0.forward(&x)?;
        py_check!(self.checker, "encoder.up_embed.out.0", &x);

        // Layer: encoder.up_embed.out.1
        x = self.encoder_up_embed_out_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_embed.out.1", &x);

        // Layer: encoder.up_embed.out.2
        x = self.encoder_up_embed_out_2.forward(&x)?;
        py_check!(self.checker, "encoder.up_embed.out.2", &x);

        // Layer: encoder.up_embed.pos_enc.dropout
        x = self.encoder_up_embed_pos_enc_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_embed.pos_enc.dropout", &x);

        // Layer: encoder.up_embed.pos_enc.dropout.1
        x = self.encoder_up_embed_pos_enc_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_embed.pos_enc.dropout.1", &x);

        // Layer: encoder.up_encoders.0.dropout
        x = self.encoder_up_encoders_0_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.dropout", &x);

        // Layer: encoder.up_encoders.0.dropout.1
        x = self.encoder_up_encoders_0_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.dropout.1", &x);

        // Layer: encoder.up_encoders.0.feed_forward.dropout
        x = self.encoder_up_encoders_0_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.feed_forward.dropout", &x);

        // Layer: encoder.up_encoders.0.feed_forward.w_1
        x = self.encoder_up_encoders_0_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.feed_forward.w_1", &x);

        // Layer: encoder.up_encoders.0.feed_forward.w_2
        x = self.encoder_up_encoders_0_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.feed_forward.w_2", &x);

        // Layer: encoder.up_encoders.0.norm_ff
        x = self.encoder_up_encoders_0_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.norm_ff", &x);

        // Layer: encoder.up_encoders.0.norm_mha
        x = self.encoder_up_encoders_0_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.norm_mha", &x);

        // Layer: encoder.up_encoders.0.self_attn.dropout
        x = self.encoder_up_encoders_0_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.dropout", &x);

        // Layer: encoder.up_encoders.0.self_attn.linear_k
        x = self.encoder_up_encoders_0_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.linear_k", &x);

        // Layer: encoder.up_encoders.0.self_attn.linear_out
        x = self.encoder_up_encoders_0_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.linear_out", &x);

        // Layer: encoder.up_encoders.0.self_attn.linear_pos
        x = self.encoder_up_encoders_0_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.linear_pos", &x);

        // Layer: encoder.up_encoders.0.self_attn.linear_q
        x = self.encoder_up_encoders_0_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.linear_q", &x);

        // Layer: encoder.up_encoders.0.self_attn.linear_v
        x = self.encoder_up_encoders_0_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.0.self_attn.linear_v", &x);

        // Layer: encoder.up_encoders.1.dropout
        x = self.encoder_up_encoders_1_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.dropout", &x);

        // Layer: encoder.up_encoders.1.dropout.1
        x = self.encoder_up_encoders_1_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.dropout.1", &x);

        // Layer: encoder.up_encoders.1.feed_forward.dropout
        x = self.encoder_up_encoders_1_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.feed_forward.dropout", &x);

        // Layer: encoder.up_encoders.1.feed_forward.w_1
        x = self.encoder_up_encoders_1_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.feed_forward.w_1", &x);

        // Layer: encoder.up_encoders.1.feed_forward.w_2
        x = self.encoder_up_encoders_1_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.feed_forward.w_2", &x);

        // Layer: encoder.up_encoders.1.norm_ff
        x = self.encoder_up_encoders_1_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.norm_ff", &x);

        // Layer: encoder.up_encoders.1.norm_mha
        x = self.encoder_up_encoders_1_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.norm_mha", &x);

        // Layer: encoder.up_encoders.1.self_attn.dropout
        x = self.encoder_up_encoders_1_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.dropout", &x);

        // Layer: encoder.up_encoders.1.self_attn.linear_k
        x = self.encoder_up_encoders_1_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.linear_k", &x);

        // Layer: encoder.up_encoders.1.self_attn.linear_out
        x = self.encoder_up_encoders_1_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.linear_out", &x);

        // Layer: encoder.up_encoders.1.self_attn.linear_pos
        x = self.encoder_up_encoders_1_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.linear_pos", &x);

        // Layer: encoder.up_encoders.1.self_attn.linear_q
        x = self.encoder_up_encoders_1_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.linear_q", &x);

        // Layer: encoder.up_encoders.1.self_attn.linear_v
        x = self.encoder_up_encoders_1_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.1.self_attn.linear_v", &x);

        // Layer: encoder.up_encoders.2.dropout
        x = self.encoder_up_encoders_2_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.dropout", &x);

        // Layer: encoder.up_encoders.2.dropout.1
        x = self.encoder_up_encoders_2_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.dropout.1", &x);

        // Layer: encoder.up_encoders.2.feed_forward.dropout
        x = self.encoder_up_encoders_2_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.feed_forward.dropout", &x);

        // Layer: encoder.up_encoders.2.feed_forward.w_1
        x = self.encoder_up_encoders_2_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.feed_forward.w_1", &x);

        // Layer: encoder.up_encoders.2.feed_forward.w_2
        x = self.encoder_up_encoders_2_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.feed_forward.w_2", &x);

        // Layer: encoder.up_encoders.2.norm_ff
        x = self.encoder_up_encoders_2_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.norm_ff", &x);

        // Layer: encoder.up_encoders.2.norm_mha
        x = self.encoder_up_encoders_2_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.norm_mha", &x);

        // Layer: encoder.up_encoders.2.self_attn.dropout
        x = self.encoder_up_encoders_2_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.dropout", &x);

        // Layer: encoder.up_encoders.2.self_attn.linear_k
        x = self.encoder_up_encoders_2_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.linear_k", &x);

        // Layer: encoder.up_encoders.2.self_attn.linear_out
        x = self.encoder_up_encoders_2_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.linear_out", &x);

        // Layer: encoder.up_encoders.2.self_attn.linear_pos
        x = self.encoder_up_encoders_2_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.linear_pos", &x);

        // Layer: encoder.up_encoders.2.self_attn.linear_q
        x = self.encoder_up_encoders_2_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.linear_q", &x);

        // Layer: encoder.up_encoders.2.self_attn.linear_v
        x = self.encoder_up_encoders_2_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.2.self_attn.linear_v", &x);

        // Layer: encoder.up_encoders.3.dropout
        x = self.encoder_up_encoders_3_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.dropout", &x);

        // Layer: encoder.up_encoders.3.dropout.1
        x = self.encoder_up_encoders_3_dropout_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.dropout.1", &x);

        // Layer: encoder.up_encoders.3.feed_forward.dropout
        x = self.encoder_up_encoders_3_feed_forward_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.feed_forward.dropout", &x);

        // Layer: encoder.up_encoders.3.feed_forward.w_1
        x = self.encoder_up_encoders_3_feed_forward_w_1.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.feed_forward.w_1", &x);

        // Layer: encoder.up_encoders.3.feed_forward.w_2
        x = self.encoder_up_encoders_3_feed_forward_w_2.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.feed_forward.w_2", &x);

        // Layer: encoder.up_encoders.3.norm_ff
        x = self.encoder_up_encoders_3_norm_ff.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.norm_ff", &x);

        // Layer: encoder.up_encoders.3.norm_mha
        x = self.encoder_up_encoders_3_norm_mha.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.norm_mha", &x);

        // Layer: encoder.up_encoders.3.self_attn.dropout
        x = self.encoder_up_encoders_3_self_attn_dropout.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.dropout", &x);

        // Layer: encoder.up_encoders.3.self_attn.linear_k
        x = self.encoder_up_encoders_3_self_attn_linear_k.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.linear_k", &x);

        // Layer: encoder.up_encoders.3.self_attn.linear_out
        x = self.encoder_up_encoders_3_self_attn_linear_out.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.linear_out", &x);

        // Layer: encoder.up_encoders.3.self_attn.linear_pos
        x = self.encoder_up_encoders_3_self_attn_linear_pos.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.linear_pos", &x);

        // Layer: encoder.up_encoders.3.self_attn.linear_q
        x = self.encoder_up_encoders_3_self_attn_linear_q.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.linear_q", &x);

        // Layer: encoder.up_encoders.3.self_attn.linear_v
        x = self.encoder_up_encoders_3_self_attn_linear_v.forward(&x)?;
        py_check!(self.checker, "encoder.up_encoders.3.self_attn.linear_v", &x);

        // Layer: encoder.up_layer.conv
        x = self.encoder_up_layer_conv.forward(&x)?;
        py_check!(self.checker, "encoder.up_layer.conv", &x);

        // Layer: encoder_proj
        x = self.encoder_proj.forward(&x)?;
        py_check!(self.checker, "encoder_proj", &x);

        // Layer: input_embedding
        x = self.input_embedding.forward(&x)?;
        py_check!(self.checker, "input_embedding", &x);

        // Layer: spk_embed_affine_layer
        x = self.spk_embed_affine_layer.forward(&x)?;
        py_check!(self.checker, "spk_embed_affine_layer", &x);

        Ok(x)
    }
    fn manual_pad(x: &Tensor, dim: usize, left: usize, right: usize) -> Result<Tensor> {
        let mut chunks = Vec::new();
        if left > 0 {
            let mut shape = x.dims().to_vec();
            shape[dim] = left;
            chunks.push(Tensor::zeros(shape, x.dtype(), x.device())?);
        }
        chunks.push(x.clone());
        if right > 0 {
            let mut shape = x.dims().to_vec();
            shape[dim] = right;
            chunks.push(Tensor::zeros(shape, x.dtype(), x.device())?);
        }
        Tensor::cat(&chunks, dim)
    }

    pub fn mean_flow(&self, tokens: &Tensor) -> Result<Tensor> {
        let _b = tokens.dim(0)?;
        let _t = tokens.dim(1)?;
        let mut x = self.input_embedding.forward(tokens)?; // [B, T, 512]

        // --- 1. encoder.embed ---
        x = self.encoder_embed_out_0.forward(&x)?;
        x = self.encoder_embed_out_1.forward(&x)?;
        x = self.encoder_embed_out_2.forward(&x)?;

        // --- 2. Pre-Lookahead ---
        {
            let residual = x.clone();
            let mut h = x.transpose(1, 2)?;
            h = Self::manual_pad(&h, 2, 0, 3)?; // Pad end
            h = self.encoder_pre_lookahead_layer_conv1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = Self::manual_pad(&h, 2, 2, 0)?; // Pad start
            h = self.encoder_pre_lookahead_layer_conv2.forward(&h)?;
            x = (h.transpose(1, 2)? + residual)?;
        }

        // --- 3. Conformer Blocks (0-5) ---
        // Block 0
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_0_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_0_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_0_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_0_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_0_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_0_norm_ff.forward(&x)?;
            h = self.encoder_encoders_0_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_0_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 1
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_1_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_1_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_1_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_1_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_1_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_1_norm_ff.forward(&x)?;
            h = self.encoder_encoders_1_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_1_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 2
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_2_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_2_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_2_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_2_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_2_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_2_norm_ff.forward(&x)?;
            h = self.encoder_encoders_2_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_2_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 3
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_3_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_3_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_3_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_3_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_3_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_3_norm_ff.forward(&x)?;
            h = self.encoder_encoders_3_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_3_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 4
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_4_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_4_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_4_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_4_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_4_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_4_norm_ff.forward(&x)?;
            h = self.encoder_encoders_4_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_4_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 5
        {
            let residual = x.clone();
            let mut h = self.encoder_encoders_5_norm_mha.forward(&x)?;
            let q = self.encoder_encoders_5_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_encoders_5_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_encoders_5_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_encoders_5_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_encoders_5_norm_ff.forward(&x)?;
            h = self.encoder_encoders_5_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_encoders_5_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }

        // --- 4. Upsample 1D ---
        {
            let current_t = x.dim(1)?;
            x = x.transpose(1, 2)?;
            x = x.unsqueeze(2)?.upsample_nearest2d(1, current_t * 2)?.squeeze(2)?;
            x = Self::manual_pad(&x, 2, 4, 0)?; // Pad beginning with 4
            x = self.encoder_up_layer_conv.forward(&x)?; // [B, 512, 2T]
            x = x.transpose(1, 2)?;
        }

        // --- 5. up_embed ---
        x = self.encoder_up_embed_out_0.forward(&x)?;
        x = self.encoder_up_embed_out_1.forward(&x)?;
        x = self.encoder_up_embed_out_2.forward(&x)?;

        // --- 6. up_encoders (0-3) ---
        // Block 0
        {
            let residual = x.clone();
            let mut h = self.encoder_up_encoders_0_norm_mha.forward(&x)?;
            let q = self.encoder_up_encoders_0_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_up_encoders_0_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_up_encoders_0_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_up_encoders_0_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_up_encoders_0_norm_ff.forward(&x)?;
            h = self.encoder_up_encoders_0_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_up_encoders_0_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 1
        {
            let residual = x.clone();
            let mut h = self.encoder_up_encoders_1_norm_mha.forward(&x)?;
            let q = self.encoder_up_encoders_1_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_up_encoders_1_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_up_encoders_1_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_up_encoders_1_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_up_encoders_1_norm_ff.forward(&x)?;
            h = self.encoder_up_encoders_1_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_up_encoders_1_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 2
        {
            let residual = x.clone();
            let mut h = self.encoder_up_encoders_2_norm_mha.forward(&x)?;
            let q = self.encoder_up_encoders_2_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_up_encoders_2_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_up_encoders_2_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_up_encoders_2_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_up_encoders_2_norm_ff.forward(&x)?;
            h = self.encoder_up_encoders_2_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_up_encoders_2_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }
        // Block 3
        {
            let residual = x.clone();
            let mut h = self.encoder_up_encoders_3_norm_mha.forward(&x)?;
            let q = self.encoder_up_encoders_3_self_attn_linear_q.forward(&h)?;
            let k = self.encoder_up_encoders_3_self_attn_linear_k.forward(&h)?;
            let v = self.encoder_up_encoders_3_self_attn_linear_v.forward(&h)?;
            h = self.apply_std_mha(&q, &k, &v, 8)?;
            x = (self.encoder_up_encoders_3_self_attn_linear_out.forward(&h)? + residual)?;

            let residual = x.clone();
            let mut h = self.encoder_up_encoders_3_norm_ff.forward(&x)?;
            h = self.encoder_up_encoders_3_feed_forward_w_1.forward(&h)?;
            h = (h.clone() * candle_nn::ops::sigmoid(&h)?)?;
            h = self.encoder_up_encoders_3_feed_forward_w_2.forward(&h)?;
            x = (h + residual)?;
        }

        // --- 7. Final ---
        x = self.encoder_after_norm.forward(&x)?;
        x = self.encoder_proj.forward(&x)?;

        Ok(x.transpose(1, 2)?) // Return [B, 80, 2T]
    }

    fn apply_std_mha(&self, q: &Tensor, k: &Tensor, v: &Tensor, n_heads: usize) -> Result<Tensor> {
        let (b, t, d) = q.dims3()?;
        let dk = d / n_heads;
        let q = q.reshape((b, t, n_heads, dk))?.transpose(1, 2)?;
        let k = k.reshape((b, t, n_heads, dk))?.transpose(1, 2)?;
        let v = v.reshape((b, t, n_heads, dk))?.transpose(1, 2)?;

        let scores = (q.matmul(&k.transpose(2, 3)?)? / (dk as f64).sqrt())?;
        let attn = candle_nn::ops::softmax(&scores, 3)?;
        let x = attn.matmul(&v)?;
        x.transpose(1, 2)?.reshape((b, t, d))
    }
}
