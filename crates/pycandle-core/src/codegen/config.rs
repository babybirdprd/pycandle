use super::types::SymbolicConfig;

impl super::Codegen {
    pub fn extract_symbolic_config(&self) -> SymbolicConfig {
        let mut dims = self.hints.clone().unwrap_or_default();

        for meta in self.manifest.values() {
            // GPT2 specific extraction
            if let Some(v) = meta.config.get("vocab_size").and_then(|v| v.as_u64()) {
                dims.entry("vocab_size".to_string()).or_insert(v as usize);
            }
            if let Some(v) = meta.config.get("n_embd").and_then(|v| v.as_u64()) {
                dims.entry("hidden_dim".to_string()).or_insert(v as usize);
            }
            if let Some(v) = meta.config.get("n_head").and_then(|v| v.as_u64()) {
                dims.entry("n_head".to_string()).or_insert(v as usize);
            }
            if let Some(v) = meta.config.get("n_layer").and_then(|v| v.as_u64()) {
                dims.entry("n_layers".to_string()).or_insert(v as usize);
            }
            if let Some(v) = meta.config.get("n_positions").and_then(|v| v.as_u64()) {
                dims.entry("context_length".to_string())
                    .or_insert(v as usize);
            }

            // Generic extraction
            match meta.module_type.as_str() {
                "Embedding" => {
                    if let Some(n) = meta.config.get("num_embeddings").and_then(|v| v.as_u64()) {
                        dims.entry("vocab_size".to_string()).or_insert(n as usize);
                    }
                    if let Some(d) = meta.config.get("embedding_dim").and_then(|v| v.as_u64()) {
                        dims.entry("hidden_dim".to_string()).or_insert(d as usize);
                    }
                }
                "Linear" | "LoRACompatibleLinear" => {
                    // If out_features is high (like 30000+), it's likely a vocab_size (lm_head)
                    if let Some(out_f) = meta.config.get("out_features").and_then(|v| v.as_u64()) {
                        if out_f > 30000 {
                            dims.entry("vocab_size".to_string())
                                .or_insert(out_f as usize);
                        }
                    }
                }
                _ => {}
            }
        }

        SymbolicConfig { dims }
    }

    pub fn render_dim(&self, value: usize, preferred_name: &str) -> String {
        // Try preferred name first
        if !preferred_name.is_empty() {
            if let Some(&v) = self.config.dims.get(preferred_name) {
                if v == value {
                    return format!("config.{}", preferred_name);
                }
            }
        }

        // Try exact value match in any dim
        for (name, &v) in &self.config.dims {
            if v == value {
                return format!("config.{}", name);
            }
        }

        value.to_string()
    }
}
