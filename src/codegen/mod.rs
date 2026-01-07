// Codegen module with helpers for different model types
pub mod gpt2;

use crate::LayerMeta;
use std::collections::HashMap;

pub struct Codegen {
    manifest: HashMap<String, LayerMeta>,
}

impl Codegen {
    pub fn new(manifest: HashMap<String, LayerMeta>) -> Self {
        Self { manifest }
    }

    pub fn generate_model_rs(&self, model_name: &str) -> String {
        let mut code = String::new();
        code.push_str("use candle_core::{Tensor, Result, Device};\n");
        code.push_str(
            "use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};\n",
        );
        code.push_str("use crate::{PyChecker, py_check};\n");

        // Add gpt2 import if GPT2 types are present
        if self.has_gpt2_types() {
            code.push_str("use crate::gpt2;\n");
        }
        code.push_str("\n");

        code.push_str(&self.generate_struct(model_name));
        code.push_str("\n\n");
        code.push_str(&self.generate_impl(model_name));

        code
    }

    fn has_gpt2_types(&self) -> bool {
        self.manifest
            .values()
            .any(|meta| gpt2::is_gpt2_type(&meta.module_type))
    }

    fn generate_struct(&self, model_name: &str) -> String {
        let mut lines = vec![format!("pub struct {} {{", model_name)];

        let mut layers: Vec<_> = self.manifest.iter().collect();
        layers.sort_by_key(|(k, _)| *k);

        for (layer_name, meta) in layers {
            if !meta.is_leaf {
                continue;
            }
            let clean_name = layer_name.replace(".", "_");
            let candle_type = self.map_type(&meta.module_type);
            lines.push(format!("    pub {}: {},", clean_name, candle_type));
        }

        lines.push("    pub checker: Option<PyChecker>,".to_string());
        lines.push("}".to_string());
        lines.join("\n")
    }

    fn map_type(&self, py_type: &str) -> String {
        // Check GPT2 helper first
        if let Some(t) = gpt2::map_type(py_type) {
            return t;
        }

        // Core types
        match py_type {
            "Linear" => "Linear".to_string(),
            "Conv1d" => "Conv1d".to_string(),
            "LayerNorm" => "LayerNorm".to_string(),
            "Embedding" => "Embedding".to_string(),
            _ => format!("() /* TODO: {} */", py_type),
        }
    }

    fn generate_impl(&self, model_name: &str) -> String {
        let mut code = format!("impl {} {{\n", model_name);

        code.push_str(
            "    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {\n",
        );

        let mut layers: Vec<_> = self.manifest.iter().collect();
        layers.sort_by_key(|(k, _)| *k);

        for (layer_name, meta) in &layers {
            if !meta.is_leaf {
                continue;
            }
            let clean_name = layer_name.replace(".", "_");
            let init_call = self.generate_init(layer_name, meta);
            code.push_str(&format!("        let {} = {};\n", clean_name, init_call));
        }

        code.push_str("\n        Ok(Self {\n");
        for (layer_name, meta) in &layers {
            if !meta.is_leaf {
                continue;
            }
            code.push_str(&format!("            {},\n", layer_name.replace(".", "_")));
        }
        code.push_str("            checker,\n");
        code.push_str("        })\n");
        code.push_str("    }\n\n");

        // Forward method
        code.push_str("    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {\n");
        code.push_str("        let mut x = xs.clone();\n");

        for (layer_name, meta) in &layers {
            if !meta.is_leaf {
                continue;
            }
            let clean_name = layer_name.replace(".", "_");
            code.push_str(&format!("\n        // Layer: {}\n", layer_name));
            code.push_str(&format!("        x = self.{}.forward(&x)?;\n", clean_name));
            code.push_str(&format!(
                "        py_check!(self.checker, \"{}\", &x);\n",
                layer_name
            ));
        }

        code.push_str("\n        Ok(x)\n");
        code.push_str("    }\n");
        code.push_str("}\n");

        code
    }

    fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
        // Check GPT2 helper first
        if let Some(init) = gpt2::generate_init(layer_name, meta) {
            return init;
        }

        // Core types
        match meta.module_type.as_str() {
            "Linear" => {
                let in_f = meta.config["in_features"].as_u64().unwrap_or(0);
                let out_f = meta.config["out_features"].as_u64().unwrap_or(0);
                let bias = meta.config["bias"].as_bool().unwrap_or(true);
                if bias {
                    format!(
                        "candle_nn::linear({}, {}, vb.pp(\"{}\"))?",
                        in_f, out_f, layer_name
                    )
                } else {
                    format!(
                        "candle_nn::linear_no_bias({}, {}, vb.pp(\"{}\"))?",
                        in_f, out_f, layer_name
                    )
                }
            }
            "Conv1d" => {
                let in_c = meta.config["in_channels"].as_u64().unwrap_or(0);
                let out_c = meta.config["out_channels"].as_u64().unwrap_or(0);
                let k = meta.config["kernel_size"].as_u64().unwrap_or(0);
                let s = meta.config["stride"].as_u64().unwrap_or(1);
                let p = meta.config["padding"].as_u64().unwrap_or(0);
                format!(
                    "candle_nn::conv1d({}, {}, {}, candle_nn::Conv1dConfig {{ stride: {}, padding: {}, ..Default::default() }}, vb.pp(\"{}\"))?",
                    in_c, out_c, k, s, p, layer_name
                )
            }
            "LayerNorm" => {
                let shape: Vec<usize> =
                    serde_json::from_value(meta.config["normalized_shape"].clone())
                        .unwrap_or_default();
                let eps = meta.config["eps"].as_f64().unwrap_or(1e-5);
                format!(
                    "candle_nn::layer_norm(vec!{:?}, candle_nn::LayerNormConfig {{ eps: {:.1e}, ..Default::default() }}, vb.pp(\"{}\"))?",
                    shape, eps, layer_name
                )
            }
            "Embedding" => {
                let n = meta.config["num_embeddings"].as_u64().unwrap_or(0);
                let d = meta.config["embedding_dim"].as_u64().unwrap_or(0);
                format!(
                    "candle_nn::embedding({}, {}, vb.pp(\"{}\"))?",
                    n, d, layer_name
                )
            }
            _ => format!(
                "todo!(\"Implement initialization for {}\")",
                meta.module_type
            ),
        }
    }
}
