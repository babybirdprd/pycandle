use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Config {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
    pub qkv_bias: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
            qkv_bias: true, // Standard GPT2 has bias
        }
    }
}

// ============================================================================
// KV Cache
// ============================================================================

#[derive(Debug, Clone)]
pub struct KVCache {
    pub k: Tensor, // (B, n_head, T, head_dim)
    pub v: Tensor, // (B, n_head, T, head_dim)
}

impl KVCache {
    pub fn new() -> Self {
        // Placeholder, usually initialized during first forward pass
        // or we use Option<KVCache>
        unimplemented!("Use Option<KVCache> for now")
    }
}

// ============================================================================
// Layers
// ============================================================================

// Conv1D in HuggingFace is actually a Linear layer with (nx, nf) weight shape
// so transposing is needed if loading from standard HF checkpoints.
// However, standard Candle `linear` expects (out, in).
// HF Conv1D weight is (in, out).
fn conv1d(in_f: usize, out_f: usize, vb: VarBuilder) -> Result<Linear> {
    let weight = vb.get((in_f, out_f), "weight")?.t()?;
    let bias = vb.get((out_f,), "bias")?;
    Ok(Linear::new(weight, Some(bias)))
}

pub struct MultiHeadAttention {
    c_attn: Linear,
    c_proj: Linear,
    n_head: usize,
    head_dim: usize,
    bias: Tensor,
    scale: f64,
}

impl MultiHeadAttention {
    pub fn new(n_embd: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let c_attn = conv1d(n_embd, 3 * n_embd, vb.pp("c_attn"))?;
        let c_proj = conv1d(n_embd, n_embd, vb.pp("c_proj"))?;
        let head_dim = n_embd / n_head;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Causal mask buffer
        let mask: Vec<_> = (0..1024)
            .flat_map(|i| (0..1024).map(move |j| if j <= i { 1.0f32 } else { 0.0f32 }))
            .collect();
        let bias = Tensor::from_vec(mask, (1, 1, 1024, 1024), vb.device())?;

        Ok(Self {
            c_attn,
            c_proj,
            n_head,
            head_dim,
            bias,
            scale,
        })
    }

    pub fn forward(&self, x: &Tensor, layer_cache: Option<&mut Option<KVCache>>) -> Result<Tensor> {
        let (b_sz, t, c) = x.dims3()?;
        let qkv = self.c_attn.forward(x)?;

        // (B, T, 3 * n_embd) -> (B, T, 3, n_head, head_dim)
        let qkv = qkv.reshape((b_sz, t, 3, self.n_head, self.head_dim))?;
        // (3, B, n_head, T, head_dim)
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;

        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;

        // KV Cache handling
        let (k, v) = if let Some(cache_opt) = layer_cache {
            if let Some(past_kv) = cache_opt.take() {
                let k = Tensor::cat(&[&past_kv.k, &k], 2)?;
                let v = Tensor::cat(&[&past_kv.v, &v], 2)?;
                *cache_opt = Some(KVCache {
                    k: k.clone(),
                    v: v.clone(),
                });
                (k, v)
            } else {
                *cache_opt = Some(KVCache {
                    k: k.clone(),
                    v: v.clone(),
                });
                (k, v)
            }
        } else {
            (k, v)
        };

        // Standard attention
        let t_total = k.dim(2)?;
        // q: (B, H, T_q, D) @ k.t: (B, H, D, T_k) -> (B, H, T_q, T_k)
        let att = (q.matmul(&k.t()?)? * self.scale)?;

        // Causal masking
        let mask = self.bias.narrow(2, 0, t)?; // Slice to current query len
        let mask = mask.narrow(3, 0, t_total)?; // Slice to total key len

        // In inference with cache, we only generate 1 token usually, but causal mask expects full sequence
        // Actually, if we are generating 1 token at pos P, we attend to all previous P tokens.
        // The mask handles the tril logic.
        // For cached inference (q len 1, k len T), we want full attention to previous.
        // The bias implementation above is static (1024,1024).
        // We should just direct mask if t > 1.
        // If t == 1 (generation), we attend to everything, no lower-triangular needed for the single row.

        let att = if t > 1 {
            let infinite =
                Tensor::new(f32::NEG_INFINITY, att.device())?.broadcast_as(att.shape())?;
            mask.eq(0.0)?.where_cond(&infinite, &att)?
        } else {
            att // Attend to all past
        };

        let att = candle_nn::ops::softmax(&att, 3)?;

        // (B, H, T_q, T_k) @ (B, H, T_k, D) -> (B, H, T_q, D)
        let y = att.matmul(&v)?;
        // (B, T_q, H, D) -> (B, T_q, C)
        let y = y.permute((0, 2, 1, 3))?.reshape((b_sz, t, c))?;

        self.c_proj.forward(&y)
    }
}

pub struct FeedForward {
    c_fc: Linear,
    c_proj: Linear,
    act: candle_nn::Activation,
}

impl FeedForward {
    pub fn new(n_embd: usize, vb: VarBuilder) -> Result<Self> {
        let c_fc = conv1d(n_embd, 4 * n_embd, vb.pp("c_fc"))?;
        let c_proj = conv1d(4 * n_embd, n_embd, vb.pp("c_proj"))?;
        Ok(Self {
            c_fc,
            c_proj,
            act: candle_nn::Activation::Gelu,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.c_fc.forward(x)?;
        let x = self.act.forward(&x)?;
        self.c_proj.forward(&x)
    }
}

pub struct TransformerBlock {
    ln_1: LayerNorm,
    attn: MultiHeadAttention,
    ln_2: LayerNorm,
    mlp: FeedForward,
}

impl TransformerBlock {
    pub fn new(n_embd: usize, n_head: usize, vb: VarBuilder) -> Result<Self> {
        let ln_1 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln_1"))?;
        let attn = MultiHeadAttention::new(n_embd, n_head, vb.pp("attn"))?;
        let ln_2 = candle_nn::layer_norm(n_embd, 1e-5, vb.pp("ln_2"))?;
        let mlp = FeedForward::new(n_embd, vb.pp("mlp"))?;
        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
        })
    }

    pub fn forward(&self, x: &Tensor, layer_cache: Option<&mut Option<KVCache>>) -> Result<Tensor> {
        let residual = x;
        let x = self.ln_1.forward(x)?;
        let x = self.attn.forward(&x, layer_cache)?;
        let x = (x + residual)?;

        let residual = &x;
        let x = self.ln_2.forward(&x)?;
        let x = self.mlp.forward(&x)?;
        let x = (x + residual)?;
        Ok(x)
    }
}

pub struct GPTModel {
    wte: Embedding,
    wpe: Embedding,
    h: Vec<TransformerBlock>,
    ln_f: LayerNorm,
}

impl GPTModel {
    pub fn new(cfg: Config, vb: VarBuilder) -> Result<Self> {
        let wte = candle_nn::embedding(cfg.vocab_size, cfg.emb_dim, vb.pp("wte"))?;
        let wpe = candle_nn::embedding(cfg.context_length, cfg.emb_dim, vb.pp("wpe"))?;

        let mut h = Vec::new();
        for i in 0..cfg.n_layers {
            h.push(TransformerBlock::new(
                cfg.emb_dim,
                cfg.n_heads,
                vb.pp(format!("h.{}", i)),
            )?);
        }

        let ln_f = candle_nn::layer_norm(cfg.emb_dim, 1e-5, vb.pp("ln_f"))?;

        Ok(Self { wte, wpe, h, ln_f })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.forward_kv(x, None)
    }

    pub fn forward_kv(
        &self,
        x: &Tensor,
        mut kv_cache: Option<&mut Vec<Option<KVCache>>>,
    ) -> Result<Tensor> {
        let (_b, t) = x.dims2()?;

        let offset = if let Some(cache) = &kv_cache {
            if let Some(Some(prev)) = cache.first() {
                prev.k.dim(2)?
            } else {
                0
            }
        } else {
            0
        };

        let pos = Tensor::arange(offset as u32, (offset + t) as u32, x.device())?;

        let tok_emb = self.wte.forward(x)?;
        let pos_emb = self.wpe.forward(&pos)?;

        let mut x = (tok_emb + pos_emb)?;

        for (i, block) in self.h.iter().enumerate() {
            let layer_cache = if let Some(cache) = &mut kv_cache {
                if cache.len() <= i {
                    cache.resize_with(i + 1, || None);
                }
                cache.get_mut(i)
            } else {
                None
            };
            x = block.forward(&x, layer_cache)?;
        }

        self.ln_f.forward(&x)
    }
}
