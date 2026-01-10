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
    mut _dropout_p: f64,
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

pub fn dim(tensor: &Tensor, dim: isize) -> Result<usize> {
    let ndim = tensor.dims().len();
    let dim = if dim < 0 { ndim as isize + dim } else { dim };
    if dim < 0 || dim >= ndim as isize {
        candle_core::bail!(
            "Dimension out of range: {} for shape {:?}",
            dim,
            tensor.shape()
        );
    }
    Ok(tensor.dims()[dim as usize])
}

pub fn masked_fill(tensor: &Tensor, mask: &Tensor, value: f64) -> Result<Tensor> {
    let on_true = Tensor::full(value, tensor.shape(), tensor.device())?.to_dtype(tensor.dtype())?;
    mask.where_cond(&on_true, tensor)
}

pub fn reshape(tensor: &Tensor, shape: &[isize]) -> Result<Tensor> {
    let mut dims = Vec::new();
    let mut infer_idx = None;
    let mut product = 1;
    for (i, &d) in shape.iter().enumerate() {
        if d == -1 {
            if infer_idx.is_some() {
                candle_core::bail!("Only one dimension can be inferred (-1)");
            }
            infer_idx = Some(i);
            dims.push(0); // placeholder
        } else {
            dims.push(d as usize);
            product *= d as usize;
        }
    }

    if let Some(idx) = infer_idx {
        let total = tensor.shape().elem_count();
        if product == 0 {
            candle_core::bail!("Cannot reshape with product 0");
        }
        if total % product != 0 {
            candle_core::bail!(
                "Cannot reshape tensor of size {} into shape {:?}",
                total,
                shape
            );
        }
        dims[idx] = total / product;
    }

    tensor.reshape(&dims[..])
}

pub fn split(tensor: &Tensor, split_size: usize, dim: isize) -> Result<Vec<Tensor>> {
    let dim = if dim < 0 {
        tensor.dims().len() as isize + dim
    } else {
        dim
    } as usize;

    let dim_size = tensor.dim(dim)?;
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < dim_size {
        let size = std::cmp::min(split_size, dim_size - start);
        chunks.push(tensor.narrow(dim, start, size)?);
        start += size;
    }
    Ok(chunks)
}

pub fn lt(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.lt(&rhs)
}

pub fn gt(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.gt(&rhs)
}

pub fn add(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.add(&rhs)
}

pub fn sub(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.sub(&rhs)
}

pub fn mul(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.mul(&rhs)
}

pub fn div(lhs: &Tensor, rhs: &Tensor) -> Result<Tensor> {
    let (lhs, rhs) = broadcast_v2(lhs, rhs)?;
    lhs.div(&rhs)
}

fn broadcast_v2(lhs: &Tensor, rhs: &Tensor) -> Result<(Tensor, Tensor)> {
    let mut lhs_shape = lhs.dims().to_vec();
    let mut rhs_shape = rhs.dims().to_vec();

    if lhs_shape == rhs_shape {
        return Ok((lhs.clone(), rhs.clone()));
    }

    let max_ndims = std::cmp::max(lhs_shape.len(), rhs_shape.len());
    while lhs_shape.len() < max_ndims {
        lhs_shape.insert(0, 1);
    }
    while rhs_shape.len() < max_ndims {
        rhs_shape.insert(0, 1);
    }

    let mut out_shape = Vec::with_capacity(max_ndims);
    for i in 0..max_ndims {
        let l = lhs_shape[i];
        let r = rhs_shape[i];
        if l == r {
            out_shape.push(l);
        } else if l == 1 {
            out_shape.push(r);
        } else if r == 1 {
            out_shape.push(l);
        } else {
            candle_core::bail!(
                "Incompatible shapes for broadcasting: {:?} and {:?}",
                lhs.shape(),
                rhs.shape()
            );
        }
    }

    let lhs = if lhs.dims() != &out_shape {
        lhs.broadcast_as(out_shape.as_slice())?
    } else {
        lhs.clone()
    };
    let rhs = if rhs.dims() != &out_shape {
        rhs.broadcast_as(out_shape.as_slice())?
    } else {
        rhs.clone()
    };

    Ok((lhs, rhs))
}

pub enum IndexItem {
    None,
    // (start, stop) - using Option for None/RangeFull logic.
    // If stop is None, it means end of dim.
    // If start is None, it means 0.
    Slice(Option<isize>, Option<isize>),
    Index(isize),
    RangeFull,
    // (start, stop, step)
    StridedSlice(Option<isize>, Option<isize>, isize),
}

pub fn index(tensor: &Tensor, args: Vec<IndexItem>) -> Result<Tensor> {
    let mut t = tensor.clone();
    let mut dim_idx = 0;

    for item in args {
        match item {
            IndexItem::None => {
                t = t.unsqueeze(dim_idx)?;
                dim_idx += 1;
            }
            IndexItem::RangeFull => {
                // Keep dimension, proceed to next
                dim_idx += 1;
            }
            IndexItem::Slice(start, stop) => {
                let dim_len = t.dim(dim_idx)?;
                let start_idx = match start {
                    Some(s) => {
                        if s < 0 {
                            (dim_len as isize + s) as usize
                        } else {
                            s as usize
                        }
                    }
                    None => 0,
                };
                let stop_idx = match stop {
                    Some(s) => {
                        if s < 0 {
                            (dim_len as isize + s) as usize
                        } else {
                            s as usize
                        }
                    }
                    None => dim_len,
                };
                let len = stop_idx.saturating_sub(start_idx);
                t = t.narrow(dim_idx, start_idx, len)?;
                dim_idx += 1;
            }
            IndexItem::StridedSlice(start, stop, step) => {
                let dim_len = t.dim(dim_idx)?;
                let start_idx = match start {
                    Some(s) => {
                        if s < 0 {
                            (dim_len as isize + s) as usize
                        } else {
                            s as usize
                        }
                    }
                    None => 0,
                };
                let stop_idx = match stop {
                    Some(s) => {
                        if s < 0 {
                            (dim_len as isize + s) as usize
                        } else {
                            s as usize
                        }
                    }
                    None => dim_len,
                };

                // Candle doesn't support stride > 1 directly in slicing
                // We must use index_select
                let indices: Vec<u32> = (start_idx..stop_idx)
                    .step_by(step as usize)
                    .map(|x| x as u32)
                    .collect();
                let idx_tensor = Tensor::new(indices.as_slice(), t.device())?;
                t = t.index_select(&idx_tensor, dim_idx)?;
                dim_idx += 1;
            }
            IndexItem::Index(i) => {
                let dim_len = t.dim(dim_idx)?;
                let idx = if i < 0 {
                    (dim_len as isize + i) as usize
                } else {
                    i as usize
                };
                t = t.narrow(dim_idx, idx, 1)?.squeeze(dim_idx)?;
                // Dimension removed, so dim_idx stays same (next dim slides in)
            }
        }
    }
    Ok(t)
}

use candle_core::D;
