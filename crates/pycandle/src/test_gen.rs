use anyhow::{Context, Result};
use pycandle_core::{GraphNode, LayerMeta};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub struct TestGenerator {
    model_name: String,
    crate_name: String,
    manifest: HashMap<String, LayerMeta>,
    graph_nodes: Vec<GraphNode>,
    project_name: String,
}

impl TestGenerator {
    pub fn new(model_name: String, manifest_path: PathBuf) -> Result<Self> {
        let manifest_content = fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;

        let full_manifest: HashMap<String, serde_json::Value> =
            serde_json::from_str(&manifest_content).context("Failed to parse manifest JSON")?;

        let manifest: HashMap<String, LayerMeta> = full_manifest
            .iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| {
                let meta: LayerMeta = serde_json::from_value(v.clone())
                    .with_context(|| format!("Failed to parse LayerMeta for {}", k))?;
                Ok((k.clone(), meta))
            })
            .collect::<Result<_>>()?;

        let graph_nodes: Vec<GraphNode> = full_manifest
            .get("_graph_nodes")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        let stem = manifest_path.file_stem().unwrap().to_str().unwrap();
        let project_name = stem.replace("_manifest", "");

        Ok(Self {
            model_name,
            crate_name: Self::detect_crate_name().unwrap_or_else(|_| "my_project".to_string()),
            manifest,
            graph_nodes,
            project_name,
        })
    }

    fn detect_crate_name() -> Result<String> {
        let content = fs::read_to_string("Cargo.toml").context("Failed to read Cargo.toml")?;
        let toml: toml::Value = toml::from_str(&content).context("Failed to parse Cargo.toml")?;

        if let Some(package) = toml.get("package") {
            if let Some(name) = package.get("name") {
                return Ok(name.as_str().unwrap_or("my_project").replace("-", "_"));
            }
        }

        Ok("my_project".to_string())
    }

    pub fn generate_test_file(&self) -> String {
        let placeholders = self.detect_placeholders();
        let inputs_code = self.generate_inputs_loading_code(&placeholders);
        let forward_call = self.generate_forward_call(&placeholders);
        let weights_path = format!(
            "{}/src/{}.safetensors",
            self.project_name, self.project_name
        );

        let mut config_code = "let config = Config {".to_string();
        for (k, v) in self.extract_config_values() {
            config_code.push_str(&format!("\n            {}: {},", k, v));
        }
        config_code.push_str("\n            ..Default::default()\n        };");

        format!(
            r#"#[cfg(test)]
mod tests {{
    use super::*;
    use candle_core::{{Device, Tensor, DType}};
    use pycandle_core::{{PyChecker, VerificationMode}};
    use anyhow::Context;
    use std::collections::HashMap;

    mod model {{
        #![allow(dead_code)]
        #![allow(unused_imports)]
        #![allow(non_snake_case)]
        #![allow(unused_variables)]
        include!("../.pycandle/generated_{}.rs"); // Point to the generated model file
    }}
    use model::{{Config, {}}};

    #[test]
    fn test_parity() -> anyhow::Result<()> {{
        // 1. Setup Device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Running on device: {{:?}}", device);

        // 2. Load Checker and Golden Trace
        let checker = PyChecker::load("{}", "{}/traces", &device)?
            .with_mode(VerificationMode::Strict);
        println!("Loaded checker with trace: {{}}", checker.name);

        // 3. Load Model
        let weights_path = "{}";
        println!("Loading weights from: {{}}", weights_path);
        let weight_map = candle_core::safetensors::load(weights_path, &device)?;
        let vb = candle_nn::VarBuilder::from_tensors(weight_map, DType::F32, &device);
        
        // 4. Load Inputs from Trace
        let trace_path = "{}/traces/{}_trace.safetensors";
        let tensors = candle_core::safetensors::load(trace_path, &device)?;
        
        {}
        let mut model = {}::load(config, vb, Some(checker.clone()))?;

{}

        // 5. Run Forward Pass & Verify
{}
        println!("✅ Parity test passed for {}!");

        Ok(())
    }}
}}
"#,
            self.model_name,   // 1. include("../.pycandle/generated_{}.rs")
            self.model_name,   // 2. Config, {}
            self.project_name, // 3. PyChecker::load("{}", ...)
            self.project_name, // 4. "{}/traces"
            weights_path,      // 5. weights_path = "{}"
            self.project_name, // 6. trace_path = "{}/..."
            self.project_name, // 7. traces/{}_trace
            config_code,       // 8. Config instantiation
            self.model_name,   // 9. mut model = {}::load(...)
            inputs_code,       // 10. inputs
            forward_call,      // 11. forward
            self.model_name    // 12. passed for {}
        )
    }

    fn detect_placeholders(&self) -> Vec<String> {
        let mut placeholders = Vec::new();
        // If we have graph nodes, use them
        if !self.graph_nodes.is_empty() {
            for node in &self.graph_nodes {
                if node.op == "placeholder" {
                    placeholders.push(node.name.clone());
                }
            }
        } else {
            // Fallback for non-FX/legacy models: assume 1 input "input"
            placeholders.push("input".to_string());
        }

        // Safety fallback: if graph exists but no placeholders found (unlikely but possible constant model)
        if placeholders.is_empty() && !self.graph_nodes.is_empty() {
            // Maybe it really has no inputs? But let's assume at least one if we are generating a test.
            // But actually, if it truly has no inputs, we shouldn't force one.
            // However mod.rs falls back to `xs: &Tensor` if placeholders.len() <= 1
            // Wait, mod.rs: if placeholders.len() <= 1 -> "xs: &Tensor".
            // So if placeholders is empty, mod.rs expects 1 argument named "xs".
            // So we must provide 1 placeholder.
            placeholders.push("xs".to_string());
        }

        placeholders
    }

    fn generate_inputs_loading_code(&self, placeholders: &[String]) -> String {
        let mut code = String::new();
        for (i, _name) in placeholders.iter().enumerate() {
            // We use standard naming convention x0, x1... for local variables
            // And load from model_input.0, model_input.1... of the trace
            code.push_str(&format!(
                "        let x{} = tensors.get(\"model_input.{}\").context(\"Missing model_input.{}\")?.clone();\n",
                i, i, i
            ));
        }
        code
    }

    fn generate_forward_call(&self, placeholders: &[String]) -> String {
        let args = if placeholders.len() <= 1 {
            // mod.rs generates `fn forward(&self, xs: &Tensor)`
            // We just pass the first input we loaded
            "&x0".to_string()
        } else {
            // mod.rs generates `fn forward(&self, xs0: &Tensor, xs1: &Tensor, ...)`
            (0..placeholders.len())
                .map(|i| format!("&x{}", i))
                .collect::<Vec<_>>()
                .join(", ")
        };

        let outputs = self.detect_outputs();
        if outputs.len() > 1 {
            let out_vars: Vec<String> = outputs
                .iter()
                .enumerate()
                .map(|(i, _)| format!("out{}", i))
                .collect();
            let mut code = format!(
                "        let ({}) = model.forward({})?;\n",
                out_vars.join(", "),
                args
            );
            for (i, name) in outputs.iter().enumerate() {
                code.push_str(&format!(
                    "        checker.verify(\"{}\", &out{})?;\n",
                    name, i
                ));
            }
            code
        } else {
            let name = outputs
                .get(0)
                .cloned()
                .unwrap_or_else(|| "output".to_string());
            format!(
                "        let output = model.forward({})?;\n        checker.verify(\"{}\", &output)?;",
                args, name
            )
        }
    }

    fn detect_outputs(&self) -> Vec<String> {
        let mut outputs = Vec::new();
        // Look for the "output" node in graph_nodes
        if let Some(node) = self.graph_nodes.iter().find(|n| n.op == "output") {
            if let Some(arg0) = node.args.get(0) {
                match arg0 {
                    serde_json::Value::String(s) => {
                        if s.contains("{") && s.contains("}") {
                            // Python dict: {'key': val, ...}
                            outputs = self.extract_values_from_dict_snippet(s);
                        } else {
                            outputs.push(s.clone());
                        }
                    }
                    serde_json::Value::Array(arr) => {
                        for val in arr {
                            if let Some(s) = val.as_str() {
                                outputs.push(s.to_string());
                            }
                        }
                    }
                    serde_json::Value::Object(obj) => {
                        for (k, _) in obj {
                            outputs.push(k.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        if outputs.is_empty() {
            // Fallback to old heuristic
            outputs.push(self.detect_output_layer());
        }
        outputs
    }

    fn extract_values_from_dict_snippet(&self, s: &str) -> Vec<String> {
        let mut keys = Vec::new();
        let content = s.trim_matches(|c| c == '{' || c == '}');
        for part in content.split(',') {
            if let Some(pos) = part.find(':') {
                let key = part[..pos].trim().trim_matches(|c| c == '\'' || c == '"');
                keys.push(key.to_string());
            }
        }
        keys
    }

    fn detect_output_layer(&self) -> String {
        // Heuristic: The layer that isn't used as an input to any other layer,
        // or the last one alphabetically if it looks like a sequence.
        // If we have FX graph, it's the last node.
        // For now, return a placeholder or try to find the "last" one.
        if let Some(last) = self.manifest.keys().max() {
            last.clone()
        } else {
            "TODO_output_layer".to_string()
        }
    }

    fn extract_config_values(&self) -> HashMap<String, usize> {
        let mut dims = HashMap::new();

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
        dims
    }
}
