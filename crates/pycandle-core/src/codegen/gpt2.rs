//! GPT2 code generation helpers
//!
//! This module provides codegen utilities for HuggingFace GPT2 models.

use crate::LayerMeta;

/// Configuration extracted from GPT2 manifest
#[derive(Debug, Clone)]
pub struct GptConfig {
    pub vocab_size: usize,
    pub context_length: usize,
    pub emb_dim: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub drop_rate: f32,
}

impl Default for GptConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            context_length: 1024,
            emb_dim: 768,
            n_heads: 12,
            n_layers: 12,
            drop_rate: 0.1,
        }
    }
}

/// Check if module type is a GPT2 variant
pub fn is_gpt2_type(module_type: &str) -> bool {
    matches!(
        module_type,
        "GPT2Model" | "GPT2LMHeadModel" | "GPT2Block" | "GPT2Attention" | "GPT2MLP"
    )
}

/// Map GPT2 Python type to Candle type
pub fn map_type(py_type: &str) -> Option<String> {
    match py_type {
        "GPT2Model" | "GPT2LMHeadModel" => Some("gpt2::GPTModel".to_string()),
        "GPT2Block" => Some("gpt2::TransformerBlock".to_string()),
        "GPT2Attention" => Some("gpt2::MultiHeadAttention".to_string()),
        "GPT2MLP" => Some("gpt2::FeedForward".to_string()),
        _ => None,
    }
}

/// Extract GPT config from layer metadata
pub fn extract_config(meta: &LayerMeta) -> GptConfig {
    GptConfig {
        vocab_size: meta
            .config
            .get("vocab_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(50257) as usize,
        context_length: meta
            .config
            .get("n_positions")
            .and_then(|v| v.as_u64())
            .unwrap_or(1024) as usize,
        emb_dim: meta
            .config
            .get("n_embd")
            .and_then(|v| v.as_u64())
            .unwrap_or(768) as usize,
        n_heads: meta
            .config
            .get("n_head")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize,
        n_layers: meta
            .config
            .get("n_layer")
            .and_then(|v| v.as_u64())
            .unwrap_or(12) as usize,
        drop_rate: meta
            .config
            .get("resid_pdrop")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.1) as f32,
    }
}

/// Generate initialization code for GPT2 layer types
pub fn generate_init(
    layer_name: &str,
    meta: &LayerMeta,
    symbolic_dims: &std::collections::HashMap<String, usize>,
) -> Option<String> {
    let render_dim = |val: usize, preferred: &str| -> String {
        if !preferred.is_empty() {
            if let Some(&v) = symbolic_dims.get(preferred) {
                if v == val {
                    return format!("config.{}", preferred);
                }
            }
        }
        for (name, &v) in symbolic_dims {
            if v == val {
                return format!("config.{}", name);
            }
        }
        val.to_string()
    };

    match meta.module_type.as_str() {
        "GPT2Model" | "GPT2LMHeadModel" => {
            let config_val = extract_config(meta);
            let vocab_size = render_dim(config_val.vocab_size, "vocab_size");
            let context_length = render_dim(config_val.context_length, "context_length");
            let emb_dim = render_dim(config_val.emb_dim, "hidden_dim");
            let n_heads = render_dim(config_val.n_heads, "n_head");
            let n_layers = render_dim(config_val.n_layers, "n_layers");

            Some(format!(
                r#"{{
                let gpt_cfg = pycandle_core::gpt2::Config {{
                    vocab_size: {},
                    context_length: {},
                    emb_dim: {},
                    n_heads: {},
                    n_layers: {},
                    drop_rate: {:.1},
                    qkv_bias: false,
                }};
                pycandle_core::gpt2::GPTModel::new(gpt_cfg, &vb.pp("{}"))?
            }}"#,
                vocab_size,
                context_length,
                emb_dim,
                n_heads,
                n_layers,
                config_val.drop_rate,
                layer_name
            ))
        }
        "GPT2Block" => Some(format!(
            "pycandle_core::gpt2::TransformerBlock::new(gpt2_cfg, &vb.pp(\"{}\"))?",
            layer_name
        )),
        "GPT2Attention" => {
            let dim_val = meta
                .config
                .get("n_embd")
                .and_then(|v| v.as_u64())
                .unwrap_or(768) as usize;
            let heads_val = meta
                .config
                .get("n_head")
                .and_then(|v| v.as_u64())
                .unwrap_or(12) as usize;
            let dim = render_dim(dim_val, "hidden_dim");
            let heads = render_dim(heads_val, "n_head");

            Some(format!(
                "pycandle_core::gpt2::MultiHeadAttention::new({}, {}, 0.1, {}, false, &vb.pp(\"{}\"))?",
                dim, dim, heads, layer_name
            ))
        }
        "GPT2MLP" => Some(format!(
            "pycandle_core::gpt2::FeedForward::new(gpt2_cfg, &vb.pp(\"{}\"))?",
            layer_name
        )),
        _ => None,
    }
}
