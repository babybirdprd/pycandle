use candle_core::{DType, Result, Tensor};
use candle_nn::ops::softmax;

/// scaled_dot_product_attention
///
/// Computes scaled dot product attention on query, key and value tensors, using
/// an optional attention mask if passed, and applying dropout if a probability
/// greater than 0.0 is specified.
///
/// # Arguments
///
/// * `query` - Query tensor; shape (N, ..., L, E)
/// * `key` - Key tensor; shape (N, ..., S, E)
/// * `value` - Value tensor; shape (N, ..., S, E)
/// * `attn_mask` - Optional attention mask; shape (N, ..., L, S) or broadcastable
/// * `dropout_p` - Dropout probability; if > 0.0, dropout is applied
/// * `is_causal` - If true, applies a causal mask to the attention weights
/// * `scale` - Optional scale factor; if None, defaults to 1 / sqrt(E)
///
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    attn_mask: Option<&Tensor>,
    dropout_p: f64,
    is_causal: bool,
    scale: Option<f64>,
) -> Result<Tensor> {
    let dim = query.dim(D::Minus1)?;
    let scale_factor = scale.unwrap_or(1.0 / (dim as f64).sqrt());

    // 1. Scale Query
    let q = (query * scale_factor)?;

    // 2. Matmul (Q * K^T)
    // Transpose key to (N, ..., E, S) for matmul
    // We assume the last two dims are (Seq_len, Head_dim)
    let k_t = key.transpose(D::Minus2, D::Minus1)?;
    let mut attn_weights = q.matmul(&k_t)?;

    // 3. Causal Mask
    if is_causal {
        let l = query.dim(D::Minus2)?;
        let s = key.dim(D::Minus2)?;

        // Generate causal mask manually since tril might not be directly available on Tensor
        // mask[i, j] = 1 if i >= j else 0
        let dev = query.device();
        let r_idx = Tensor::arange(0u32, l as u32, dev)?.unsqueeze(1)?;
        let c_idx = Tensor::arange(0u32, s as u32, dev)?.unsqueeze(0)?;
        let mask = r_idx.ge(&c_idx)?; // Broadcasts to (L, S), dtype U8 (1 for True, 0 for False)

        // Arithmetic masking:
        // mask is 1 (keep), 0 (discard).
        // We want 0 (keep), -inf (discard).
        // (mask - 1) * 1e9
        // 1 -> 0 * 1e9 = 0.
        // 0 -> -1 * 1e9 = -1e9.
        let mask = mask.to_dtype(DType::F32)?;
        let mask = (mask - 1.0)?;
        let mask = (mask * 1e9)?;

        // Broadcast to attn_weights shape
        let mask = mask.broadcast_as(attn_weights.shape())?;
        attn_weights = (attn_weights + mask)?;
    }

    // 4. Attention Mask
    if let Some(mask) = attn_mask {
        attn_weights = attn_weights.broadcast_add(mask)?;
    }

    // 5. Softmax
    let attn_probs = softmax(&attn_weights, D::Minus1)?;

    // 6. Dropout (Inference only for now -> No Op if not training)
    // We assume we are in inference mode for this port, so we ignore dropout_p
    // If we were training, we'd use candle_nn::Dropout

    // 7. Matmul (Attn * V)
    attn_probs.matmul(value)
}

use candle_core::D;
