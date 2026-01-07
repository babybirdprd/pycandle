---
description: Add GPT2 transformer support to PyCandle codegen using gpt2.rs reference
---

# Add GPT2 Support to PyCandle

Use the reference `gpt2.rs` implementation to add GPT2 transformer support to PyCandle codegen.

## Reference Implementation

The file `gpt2.rs` contains a complete Candle GPT2 implementation:
- `GPTModel` - Full model with token/position embeddings
- `TransformerBlock` - Attention + FFN with residual connections
- `MultiHeadAttention` - Scaled dot-product attention
- `FeedForward` - MLP with GELU activation
- `LayerNorm` - Custom layer normalization
- `GELU` - Gaussian Error Linear Unit activation

## Steps

### 1. Update spy.py to extract transformer configs

Add detection for HuggingFace GPT2 models:

```python
def _get_module_config(self, m: nn.Module) -> Dict[str, Any]:
    config = {}
    # ... existing code ...
    
    # GPT2 from transformers
    if hasattr(m, 'config') and hasattr(m.config, 'n_embd'):
        config['vocab_size'] = m.config.vocab_size
        config['n_positions'] = m.config.n_positions  # context_length
        config['n_embd'] = m.config.n_embd  # emb_dim
        config['n_head'] = m.config.n_head  # n_heads
        config['n_layer'] = m.config.n_layer  # n_layers
        config['resid_pdrop'] = m.config.resid_pdrop  # drop_rate
    
    return config
```

### 2. Add GPT2 module mappings to codegen.rs

```rust
fn map_type(&self, py_type: &str) -> String {
    match py_type {
        // ... existing ...
        "GPT2Model" => "GPTModel".to_string(),
        "GPT2LMHeadModel" => "GPTModel".to_string(),
        "GPT2Block" => "TransformerBlock".to_string(),
        "GPT2Attention" => "MultiHeadAttention".to_string(),
        "GPT2MLP" => "FeedForward".to_string(),
        _ => format!("() /* TODO: {} */", py_type),
    }
}
```

### 3. Generate GPT2 initialization code

```rust
fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
    match meta.module_type.as_str() {
        "GPT2Model" | "GPT2LMHeadModel" => {
            let cfg = self.extract_gpt_config(meta);
            format!(r#"{{
                let cfg = gpt2::Config {{
                    vocab_size: {},
                    context_length: {},
                    emb_dim: {},
                    n_heads: {},
                    n_layers: {},
                    drop_rate: {},
                    qkv_bias: false,
                }};
                gpt2::GPTModel::new(cfg, &vb.pp("{}"))?
            }}"#, cfg.vocab_size, cfg.context_length, cfg.emb_dim, 
                cfg.n_heads, cfg.n_layers, cfg.drop_rate, layer_name)
        }
        "GPT2Block" => {
            format!("gpt2::TransformerBlock::new(cfg, &vb.pp(\"{}\"))?", layer_name)
        }
        "GPT2Attention" => {
            let dim = meta.config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(768) as usize;
            let heads = meta.config.get("num_attention_heads").and_then(|v| v.as_u64()).unwrap_or(12) as usize;
            format!("gpt2::MultiHeadAttention::new({}, {}, 0.1, {}, false, &vb.pp(\"{}\"))?", 
                dim, dim, heads, layer_name)
        }
        // ... other cases
    }
}

fn extract_gpt_config(&self, meta: &LayerMeta) -> GptConfig {
    GptConfig {
        vocab_size: meta.config.get("vocab_size").and_then(|v| v.as_u64()).unwrap_or(50257) as usize,
        context_length: meta.config.get("n_positions").and_then(|v| v.as_u64()).unwrap_or(1024) as usize,
        emb_dim: meta.config.get("n_embd").and_then(|v| v.as_u64()).unwrap_or(768) as usize,
        n_heads: meta.config.get("n_head").and_then(|v| v.as_u64()).unwrap_or(12) as usize,
        n_layers: meta.config.get("n_layer").and_then(|v| v.as_u64()).unwrap_or(12) as usize,
        drop_rate: meta.config.get("resid_pdrop").and_then(|v| v.as_f64()).unwrap_or(0.1) as f32,
    }
}
```

### 4. Add gpt2 module to lib.rs

```rust
// src/lib.rs
pub mod gpt2;

// Re-export for convenience
pub use gpt2::{GPTModel, Config as GPTConfig, TransformerBlock, MultiHeadAttention};
```

### 5. Update generated code template

When generating for GPT2-based models, include the config struct:

```rust
// In generated code header
use crate::gpt2::{self, GPTModel, Config as GPTConfig};

impl TurboT3Components {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        // GPT2 backbone with Turbo config
        let gpt_cfg = GPTConfig {
            vocab_size: 50276,
            context_length: 1024,
            emb_dim: 1024,
            n_heads: 16,
            n_layers: 24,
            drop_rate: 0.1,
            qkv_bias: false,
        };
        let backbone = GPTModel::new(gpt_cfg, &vb.pp("tfmr"))?;
        // ...
    }
}
```

## Weight Mapping

HuggingFace GPT2 weights need mapping to our structure:

| HuggingFace Key | Our Key |
|-----------------|---------|
| `wte.weight` | `token_emb.weight` |
| `wpe.weight` | `pos_emb.weight` |
| `h.{i}.ln_1.*` | `transformer_block_{i}.layer_norm1.*` |
| `h.{i}.ln_2.*` | `transformer_block_{i}.layer_norm2.*` |
| `h.{i}.attn.c_attn.weight` | Split into `w_query`, `w_key`, `w_value` |
| `h.{i}.attn.c_proj.*` | `transformer_block_{i}.mha.out_proj.*` |
| `h.{i}.mlp.c_fc.*` | `transformer_block_{i}.ff.ff_top.*` |
| `h.{i}.mlp.c_proj.*` | `transformer_block_{i}.ff.ff_bottom.*` |
| `ln_f.*` | `final_layer_norm.*` |

### Weight conversion helper

```rust
// src/gpt2_weights.rs
pub fn convert_hf_weights(hf_tensors: &HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>> {
    let mut converted = HashMap::new();
    
    // Token and position embeddings
    if let Some(wte) = hf_tensors.get("wte.weight") {
        converted.insert("token_emb.weight".to_string(), wte.clone());
    }
    if let Some(wpe) = hf_tensors.get("wpe.weight") {
        converted.insert("pos_emb.weight".to_string(), wpe.clone());
    }
    
    // Transformer blocks
    for i in 0..24 {  // Adjust based on n_layers
        let prefix = format!("h.{}", i);
        let new_prefix = format!("transformer_block_{}", i);
        
        // Split c_attn into Q, K, V
        if let Some(c_attn) = hf_tensors.get(&format!("{}.attn.c_attn.weight", prefix)) {
            let chunks = c_attn.chunk(3, 1)?;
            converted.insert(format!("{}.mha.w_query.weight", new_prefix), chunks[0].clone());
            converted.insert(format!("{}.mha.w_key.weight", new_prefix), chunks[1].clone());
            converted.insert(format!("{}.mha.w_value.weight", new_prefix), chunks[2].clone());
        }
        // ... continue for other weights
    }
    
    Ok(converted)
}
```

## Testing

```bash
# 1. Record a GPT2 model from HuggingFace
cd chatterbox-repo
cat > record_gpt2.py << 'EOF'
import sys
sys.path.insert(0, "../py")
from spy import GoldenRecorder
from transformers import GPT2LMHeadModel

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

import torch
input_ids = torch.randint(0, 50257, (1, 32))

recorder = GoldenRecorder(output_dir="py_trace")
recorder.record(model, input_ids)
recorder.save("gpt2_hf")
EOF

.\.venv\Scripts\python record_gpt2.py

# 2. Generate Candle code
cargo run -- codegen --manifest py_trace/gpt2_hf_manifest.json --out generated_gpt2.rs --model GPT2

# 3. Verify GPT2 layers are properly mapped (not todo!())
cargo run -- todos --file generated_gpt2.rs --json | jq '.total'
```
