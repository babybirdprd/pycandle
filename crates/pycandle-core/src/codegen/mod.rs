//! Code generation from PyTorch manifests to Candle Rust code
//!
//! This module generates idiomatic Rust code from recorded PyTorch model manifests.

pub mod gpt2;

use crate::LayerMeta;
use serde::Serialize;
use std::collections::HashMap;

// ============================================================================
// Internal Types for Codegen
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReturnType {
    Tensor,
    Tuple,
    Vec,
    Primitive,
}

// ============================================================================
// JSON Output Structs for Analysis
// ============================================================================

#[derive(Serialize, Debug)]
pub struct AnalysisResult {
    pub supported: usize,
    pub unsupported: usize,
    pub total: usize,
    pub coverage_percent: f32,
    pub gaps: Vec<GapInfo>,
    pub layers: Vec<LayerInfo>,
}

#[derive(Serialize, Debug)]
pub struct GapInfo {
    pub module_type: String,
    pub count: usize,
    pub suggestion: String,
}

#[derive(Serialize, serde::Deserialize, Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub module_type: String,
    pub supported: bool,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(serde::Deserialize, Debug, Clone)]
pub struct GraphNode {
    pub name: String,
    pub op: String,
    pub target: String,
    pub args: Vec<serde_json::Value>,
    #[serde(default)]
    pub kwargs: HashMap<String, serde_json::Value>,
    pub module_type: Option<String>,
}

#[derive(Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct SymbolicConfig {
    pub dims: HashMap<String, usize>,
}

/// Code generator that converts manifests to Rust code
pub struct Codegen {
    pub manifest: HashMap<String, LayerMeta>,
    pub hints: Option<HashMap<String, usize>>,
    pub graph_nodes: Vec<GraphNode>,
    pub stateful: bool,
    config: SymbolicConfig,
}

impl Codegen {
    pub fn new(
        manifest: HashMap<String, LayerMeta>,
        hints: Option<HashMap<String, usize>>,
    ) -> Self {
        let mut slf = Self {
            manifest,
            hints: hints.clone(),
            graph_nodes: Vec::new(),
            stateful: false,
            config: SymbolicConfig::default(),
        };
        slf.config = slf.extract_symbolic_config();
        slf
    }

    pub fn with_graph(mut self, graph_nodes: Vec<GraphNode>) -> Self {
        self.graph_nodes = graph_nodes;
        self
    }

    pub fn with_stateful(mut self, stateful: bool) -> Self {
        self.stateful = stateful;
        self
    }

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

    fn render_dim(&self, value: usize, preferred_name: &str) -> String {
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

    fn sanitize_name(&self, name: &str) -> String {
        // Handle ONNX paths like /layers/0/Gather_1_output -> gather_1
        if name.contains('/') {
            let clean = name.trim_start_matches('/');
            let parts: Vec<&str> = clean.split('/').collect();
            let last = parts.last().unwrap_or(&name);
            let mut sanitized = last.to_lowercase().replace("_output", "");

            if sanitized.starts_with("node_") {
                sanitized = sanitized.replace("node_", "x_");
            }

            // Ensure valid identifier
            if sanitized
                .chars()
                .next()
                .map(|c| !c.is_alphabetic())
                .unwrap_or(true)
            {
                return format!("x_{}", sanitized);
            }
            return sanitized;
        }

        // Standard PyTorch names: encoder.layers.0 -> encoder_layers_0
        let mut sanitized = name.replace(".", "_").replace("-", "_");
        if sanitized.starts_with("node_") {
            sanitized = sanitized.replace("node_", "x_");
        }

        // Ensure valid identifier
        if sanitized
            .chars()
            .next()
            .map(|c| !c.is_alphabetic() && c != '_')
            .unwrap_or(true)
        {
            return format!("x_{}", sanitized);
        }

        sanitized
    }

    /// Analyze the manifest and return a structured result for JSON output
    pub fn analyze(&self) -> AnalysisResult {
        let mut supported = 0;
        let mut unsupported = 0;
        let mut gap_counts: HashMap<String, usize> = HashMap::new();
        let mut layers = Vec::new();

        for (name, meta) in &self.manifest {
            if !meta.is_leaf {
                continue;
            }

            let is_supported = self.is_supported(&meta.module_type);
            if is_supported {
                supported += 1;
            } else {
                unsupported += 1;
                *gap_counts.entry(meta.module_type.clone()).or_default() += 1;
            }

            layers.push(LayerInfo {
                name: name.clone(),
                module_type: meta.module_type.clone(),
                supported: is_supported,
                input_shapes: meta.input_shapes.clone(),
                output_shapes: meta.output_shapes.clone(),
            });
        }

        // Sort layers by name for consistent output
        layers.sort_by(|a, b| a.name.cmp(&b.name));

        let gaps: Vec<GapInfo> = gap_counts
            .into_iter()
            .map(|(t, c)| GapInfo {
                suggestion: self.get_suggestion(&t),
                module_type: t,
                count: c,
            })
            .collect();

        let total = supported + unsupported;
        let coverage_percent = if total > 0 {
            (supported as f32 / total as f32) * 100.0
        } else {
            100.0
        };

        AnalysisResult {
            supported,
            unsupported,
            total,
            coverage_percent,
            gaps,
            layers,
        }
    }

    /// Check if a module type is supported by the codegen
    pub fn is_supported(&self, module_type: &str) -> bool {
        // Check GPT2 helper first
        if gpt2::map_type(module_type).is_some() {
            return true;
        }

        matches!(
            module_type,
            "Linear"
                | "Conv1d"
                | "LayerNorm"
                | "Embedding"
                | "ReLU"
                | "GELU"
                | "Sigmoid"
                | "Tanh"
                | "ELU"
                | "LeakyReLU"
                | "Snake"
                | "BatchNorm1d"
                | "BatchNorm2d"
                | "LSTM"
                | "Mish"
                | "SiLU"
                | "CausalConv1d"
                | "Transpose"
                | "Conv1D"
                | "Dropout"
                | "NewGELUActivation"
                | "LoRACompatibleLinear"
                | "SinusoidalPosEmb"
        )
    }

    /// Get implementation suggestion for an unsupported module type
    pub fn get_suggestion(&self, module_type: &str) -> String {
        match module_type {
            "LSTM" => "Use /add-lstm workflow".to_string(),
            "BatchNorm1d" | "BatchNorm2d" => "Use /add-batchnorm workflow".to_string(),
            "Snake" | "ELU" => "Use /add-activations workflow".to_string(),
            _ => format!("Implement {} manually", module_type),
        }
    }

    pub fn generate_model_rs(&self, model_name: &str) -> String {
        let mut code = String::new();
        code.push_str("use candle_core::{Result, Tensor, IndexOp, Shape};\n");
        code.push_str("use candle_nn::{Module, VarBuilder};\n");
        code.push_str("use pycandle_core::{PyChecker, py_check, VerificationMode, layers::*};\n\n");

        if self.stateful {
            code.push_str(
                r#"#[derive(Debug, Clone)]
pub struct KVCache {
    pub k: Tensor,
    pub v: Tensor,
}

"#,
            );
        }

        code.push_str(&self.generate_config_struct());
        code.push_str("\n");
        code.push_str(&self.generate_struct(model_name));
        code.push_str("\n");
        code.push_str(&self.generate_impl(model_name));

        code
    }

    fn generate_config_struct(&self) -> String {
        let mut lines = vec!["pub struct Config {".to_string()];
        let mut dims: Vec<_> = self.config.dims.iter().collect();
        dims.sort_by_key(|(k, _)| *k);

        for (name, value) in dims {
            lines.push(format!("    pub {}: usize, // {}", name, value));
        }
        lines.push("}".to_string());
        lines.join("\n")
    }

    fn has_gpt2_types(&self) -> bool {
        self.manifest
            .values()
            .any(|meta| gpt2::is_gpt2_type(&meta.module_type))
    }

    pub fn generate_struct(&self, model_name: &str) -> String {
        let mut code = format!("pub struct {} {{\n", model_name);

        let mut sorted_keys: Vec<_> = self.manifest.keys().collect();
        sorted_keys.sort();

        for name in sorted_keys {
            let meta = &self.manifest[name];
            if meta.is_leaf {
                let rust_type = self.map_type(&meta.module_type);
                code.push_str(&format!(
                    "    pub {}: {},\n",
                    self.sanitize_name(name),
                    rust_type
                ));
            }
        }

        code.push_str("    pub checker: Option<PyChecker>,\n");
        if self.stateful {
            code.push_str("    pub cache: std::cell::RefCell<Vec<Option<KVCache>>>,\n");
        }
        code.push_str("}\n");
        code
    }

    fn map_type(&self, py_type: &str) -> String {
        // Check GPT2 helper first
        if let Some(t) = gpt2::map_type(py_type) {
            return t;
        }

        // Core types
        match py_type {
            "Linear" => "candle_nn::Linear".to_string(),
            "Conv1d" => "candle_nn::Conv1d".to_string(),
            "LayerNorm" => "candle_nn::LayerNorm".to_string(),
            "Embedding" => "candle_nn::Embedding".to_string(),
            // Activations
            "ReLU" => "ReLU".to_string(),
            "GELU" => "GELU".to_string(),
            "Sigmoid" => "Sigmoid".to_string(),
            "Tanh" => "Tanh".to_string(),
            "ELU" => "ELU".to_string(),
            "LeakyReLU" => "LeakyReLU".to_string(),
            "Snake" => "Snake".to_string(),
            "BatchNorm1d" => "BatchNorm1d".to_string(),
            "BatchNorm2d" => "BatchNorm2d".to_string(),
            "LSTM" => "LSTM".to_string(),
            "Mish" => "Mish".to_string(),
            "SiLU" => "SiLU".to_string(),
            "CausalConv1d" => "CausalConv1d".to_string(),
            "Transpose" => "Transpose".to_string(),
            "Conv1D" => "candle_nn::Linear".to_string(),
            "Dropout" => "Dropout".to_string(),
            "NewGELUActivation" => "candle_nn::Activation".to_string(),
            "LoRACompatibleLinear" => "candle_nn::Linear".to_string(),
            "SinusoidalPosEmb" => "SinusoidalPosEmb".to_string(),
            _ => format!("() /* TODO: {} */", py_type),
        }
    }

    pub fn generate_impl(&self, model_name: &str) -> String {
        let mut code = format!("impl {} {{\n", model_name);

        // Load method
        code.push_str("    #[allow(unused_variables)]\n");
        code.push_str(&format!(
            "    pub fn load(config: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {{\n"
        ));

        if self.has_gpt2_types() {
            code.push_str(
                r#"        let gpt2_cfg = pycandle_core::gpt2::Config {
            vocab_size: config.vocab_size,
            context_length: config.context_length,
            emb_dim: config.hidden_dim,
            n_heads: config.n_head,
            n_layers: config.n_layers,
            ..Default::default()
        };
"#,
            );
        }

        let mut sorted_keys: Vec<_> = self.manifest.keys().collect();
        sorted_keys.sort();

        for name in &sorted_keys {
            let meta = self.manifest.get(*name).unwrap();
            if meta.is_leaf {
                let field = self.sanitize_name(name);
                let init = self.generate_init(name, meta);
                code.push_str(&format!("        let {} = {};\n", field, init));
            }
        }

        let mut fields = vec![];
        for name in &sorted_keys {
            if self.manifest.get(*name).unwrap().is_leaf {
                fields.push(self.sanitize_name(name));
            }
        }
        fields.push("checker".to_string());

        code.push_str(&format!("        Ok(Self {{ {}", fields.join(", ")));
        if self.stateful {
            code.push_str(", cache: std::cell::RefCell::new(Vec::new())");
        }
        code.push_str(" })\n");
        code.push_str("    }\n\n");

        // Forward methods
        if self.graph_nodes.is_empty() {
            // Sequential fallback
            code.push_str("    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {\n");
            if self.stateful {
                code.push_str(
                    "        self.forward_with_cache(xs, &mut self.cache.borrow_mut())\n",
                );
                code.push_str("    }\n\n");
                code.push_str("    pub fn forward_with_cache(&self, xs: &Tensor, cache: &mut Vec<Option<KVCache>>) -> Result<Tensor> {\n");
            }
            code.push_str("        let mut x = xs.clone();\n");

            for name in &sorted_keys {
                let meta = self.manifest.get(*name).unwrap();
                if !meta.is_leaf {
                    continue;
                }
                let clean_name = self.sanitize_name(name);
                let forward_call = if self.map_type(&meta.module_type) == "LSTM" {
                    format!("self.{}.forward(&x)?.0", clean_name)
                } else {
                    format!("self.{}.forward(&x)?", clean_name)
                };
                code.push_str(&format!("\n        // Layer: {}\n", name));
                code.push_str(&format!("        x = {};\n", forward_call));
                code.push_str(&format!(
                    "        py_check!(self.checker, \"{}\", &x);\n",
                    name
                ));
            }

            code.push_str("\n        Ok(x)\n");
            code.push_str("    }\n");
        } else {
            // DAG / torch.fx based forward
            code.push_str(&self.generate_forward_dag(&self.graph_nodes));
        }
        code.push_str("}\n");

        code
    }

    fn generate_forward_dag(&self, nodes: &[GraphNode]) -> String {
        let mut placeholders = Vec::new();
        for node in nodes {
            if node.op == "placeholder" {
                placeholders.push(node.name.clone());
            }
        }

        let mut code = String::new();
        let ret_type = self.get_forward_return_type();

        let inputs = if placeholders.len() <= 1 {
            "xs: &Tensor".to_string()
        } else {
            placeholders
                .iter()
                .enumerate()
                .map(|(i, _)| format!("xs{}: &Tensor", i))
                .collect::<Vec<_>>()
                .join(", ")
        };

        code.push_str(&format!(
            "    pub fn forward(&self, {}) -> {} {{\n",
            inputs, ret_type
        ));

        // Map python placeholder names to Rust input variables
        let mut var_map = HashMap::new();
        let mut node_types = HashMap::new();

        let mut placeholder_idx = 0;
        for node in nodes {
            match node.op.as_str() {
                "placeholder" => {
                    let var_name = if placeholders.len() <= 1 {
                        "xs".to_string()
                    } else {
                        format!("xs{}", placeholder_idx)
                    };
                    var_map.insert(node.name.clone(), var_name);
                    node_types.insert(node.name.clone(), ReturnType::Tensor);
                    placeholder_idx += 1;
                }
                "call_module" => {
                    let clean_name = self.sanitize_name(&node.target);
                    let var_name = self.sanitize_name(&node.name);
                    let input_var = if let Some(arg0) = node.args.get(0) {
                        match arg0 {
                            serde_json::Value::String(s) => {
                                var_map.get(s).cloned().unwrap_or(s.clone())
                            }
                            _ => {
                                let first_p = placeholders.get(0).and_then(|p| var_map.get(p));
                                first_p.cloned().unwrap_or("xs".to_string())
                            }
                        }
                    } else {
                        let first_p = placeholders.get(0).and_then(|p| var_map.get(p));
                        first_p.cloned().unwrap_or("xs".to_string())
                    };

                    let module_type = node.module_type.clone().unwrap_or_default();
                    let is_lstm = self.map_type(&module_type) == "LSTM";

                    let (forward_call, return_type) = if is_lstm {
                        // LSTM returns (output, (h, c))
                        (
                            format!("self.{}.forward(&{})?", clean_name, input_var),
                            ReturnType::Tuple,
                        )
                    } else {
                        let meta = self.manifest.get(&node.target);
                        let multi = meta.map(|m| m.output_shapes.len() > 1).unwrap_or(false);
                        if multi {
                            (
                                format!("self.{}.forward(&{})?", clean_name, input_var),
                                ReturnType::Tuple,
                            )
                        } else {
                            (
                                format!("self.{}.forward(&{})?", clean_name, input_var),
                                ReturnType::Tensor,
                            )
                        }
                    };

                    code.push_str(&format!("        let {} = {};\n", var_name, forward_call));

                    // Only do parity check on single Tensors for now to avoid tuple mismatch in py_check!
                    if return_type == ReturnType::Tensor {
                        code.push_str(&format!(
                            "        py_check!(self.checker, \"{}\", &{});\n",
                            node.target, var_name
                        ));
                    }

                    var_map.insert(node.name.clone(), var_name.clone());
                    node_types.insert(node.name.clone(), return_type);
                }
                "call_function" => {
                    let var_name = self.sanitize_name(&node.name);
                    let mut resolved_args = Vec::new();
                    for arg in &node.args {
                        // Strictly resolve args, handling literals
                        resolved_args.push(self.resolve_fx_arg(arg, &var_map));
                    }

                    // Pass node_types to map_fx_op for type-aware generation
                    let (expr, return_type) = self.map_fx_op(
                        &node.target,
                        &resolved_args,
                        &var_map,
                        &node_types,
                        &node.args,
                        &node.kwargs,
                    );
                    code.push_str(&format!("        let {} = {};\n", var_name, expr));
                    var_map.insert(node.name.clone(), var_name);
                    node_types.insert(node.name.clone(), return_type);
                }
                "call_method" => {
                    let var_name = self.sanitize_name(&node.name);
                    let mut resolved_args = Vec::new();
                    for arg in &node.args {
                        resolved_args.push(self.resolve_fx_arg(arg, &var_map));
                    }

                    if !resolved_args.is_empty() {
                        let self_var_name = match &node.args[0] {
                            serde_json::Value::String(s) => s.clone(),
                            _ => "".to_string(),
                        };
                        let self_var = &resolved_args[0];
                        let method_args = &resolved_args[1..];
                        let (expr, return_type) = self.map_fx_method(
                            &node.target,
                            self_var,
                            &self_var_name,
                            method_args,
                            &node_types,
                        );
                        code.push_str(&format!("        let {} = {};\n", var_name, expr));
                        var_map.insert(node.name.clone(), var_name);
                        node_types.insert(node.name.clone(), return_type);
                    }
                }
                "output" => {
                    let out_var = if let Some(arg0) = node.args.get(0) {
                        match arg0 {
                            serde_json::Value::String(s) => {
                                var_map.get(s).cloned().unwrap_or(s.clone())
                            }
                            serde_json::Value::Array(arr) => {
                                let items: Vec<String> = arr
                                    .iter()
                                    .map(|v| self.resolve_fx_arg(v, &var_map))
                                    .collect();
                                format!("({})", items.join(", "))
                            }
                            _ => {
                                let first_p = placeholders.get(0).and_then(|p| var_map.get(p));
                                first_p.cloned().unwrap_or("xs".to_string())
                            }
                        }
                    } else {
                        let first_p = placeholders.get(0).and_then(|p| var_map.get(p));
                        first_p.cloned().unwrap_or("xs".to_string())
                    };
                    code.push_str(&format!("        Ok({})\n", out_var));
                }
                _ => {}
            }
        }

        code.push_str("    }\n");
        code
    }

    fn resolve_fx_arg(&self, arg: &serde_json::Value, var_map: &HashMap<String, String>) -> String {
        match arg {
            serde_json::Value::String(s) => var_map.get(s).cloned().unwrap_or(s.clone()),
            serde_json::Value::Number(n) => n.to_string(), // Literal numbers
            serde_json::Value::Bool(b) => b.to_string(),   // Literal bools
            serde_json::Value::Array(arr) => {
                let items: Vec<String> = arr
                    .iter()
                    .map(|v| self.resolve_fx_arg(v, var_map))
                    .collect();
                format!("&[{}]", items.join(", "))
            }
            _ => arg.to_string(),
        }
    }

    fn map_fx_op(
        &self,
        target: &str,
        args: &[String],
        var_map: &HashMap<String, String>,
        node_types: &HashMap<String, ReturnType>,
        _raw_args: &[serde_json::Value], // Needed to check for literals
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> (String, ReturnType) {
        let target_lower = target.to_lowercase();

        // Common binary ops
        if target_lower.contains("add") && args.len() >= 2 {
            return (
                format!("(&{} + &{})?", args[0], args[1]),
                ReturnType::Tensor,
            );
        }
        if target_lower.contains("sub") && args.len() >= 2 {
            return (
                format!("(&{} - &{})?", args[0], args[1]),
                ReturnType::Tensor,
            );
        }
        if target_lower.contains("mul") && args.len() >= 2 {
            return (
                format!("(&{} * &{})?", args[0], args[1]),
                ReturnType::Tensor,
            );
        }
        if (target_lower.contains("div") || target_lower.contains("truediv")) && args.len() >= 2 {
            return (
                format!("(&{} / &{})?", args[0], args[1]),
                ReturnType::Tensor,
            );
        }

        // Robust loose matching for tricky ops
        if target.ends_with("getitem") {
            let idx = &args[1];
            // Find the source variable name to check its type
            let src_name = var_map
                .iter()
                .find(|(_, v)| *v == &args[0])
                .map(|(k, _)| k.as_str())
                .unwrap_or(&args[0]);

            let src_type = node_types
                .get(src_name)
                .cloned()
                .unwrap_or(ReturnType::Tensor);

            match src_type {
                ReturnType::Tuple => {
                    // Tuple indexing: x.0, x.1
                    if let Ok(i) = idx.parse::<usize>() {
                        return (format!("{}.{}", args[0], i), ReturnType::Tensor);
                    } else {
                        return (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor);
                    }
                }
                ReturnType::Vec => {
                    // Vec indexing: x[0] or x[-1]
                    if let Ok(i) = idx.parse::<isize>() {
                        if i < 0 {
                            let offset = i.abs();
                            return (
                                format!("{}[{}.len() - {}].clone()", args[0], args[0], offset),
                                ReturnType::Primitive,
                            );
                        } else {
                            return (format!("{}[{}].clone()", args[0], i), ReturnType::Primitive);
                        }
                    } else {
                        return (
                            format!("{}[{}].clone()", args[0], args[1]),
                            ReturnType::Primitive,
                        );
                    }
                }
                ReturnType::Tensor => {
                    if idx.contains("slice(") || idx == "None" || idx.starts_with("&[") {
                        // Map to .i() for indexing/slicing
                        let mut cleaned = idx.clone();
                        if cleaned.starts_with("&[") {
                            cleaned = cleaned[2..cleaned.len() - 1].to_string();
                        }

                        let items: Vec<String> = cleaned
                            .split(",")
                            .enumerate()
                            .map(|(i, s)| self.parse_slice_item(s.trim(), &args[0], i))
                            .collect();

                        let final_idx = if items.len() == 1 {
                            items[0].clone()
                        } else {
                            format!("({})", items.join(", "))
                        };

                        return (format!("{}.i({})?", args[0], final_idx), ReturnType::Tensor);
                    } else {
                        return (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor);
                    }
                }
                ReturnType::Primitive => {
                    return (
                        format!("todo!(/* primitive indexing on {} */)", args[0]),
                        ReturnType::Primitive,
                    );
                }
            }
        }

        // Functional ops
        match target {
            "torch.cat" | "cat" => {
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                let mut tensors = args[0].clone();
                if tensors.starts_with("&[") {
                    // Convert &[x, y] to &[&x, &y] for Candle's cat
                    let content = &tensors[2..tensors.len() - 1];
                    let items: Vec<String> =
                        content.split(", ").map(|s| format!("&{}", s)).collect();
                    tensors = format!("&[{}]", items.join(", "));
                }
                (
                    format!("Tensor::cat({}, {})?", tensors, dim),
                    ReturnType::Tensor,
                )
            }
            "torch.chunk" | "chunk" => {
                let chunks = args.get(1).map(|s| s.as_str()).unwrap_or("2");
                let dim = args.get(2).map(|s| s.as_str()).unwrap_or("0");
                (
                    format!("{}.chunk({}, {})?", args[0], chunks, dim),
                    ReturnType::Vec,
                )
            }
            "torch.split" | "split" => {
                let split_size = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                let dim = args.get(2).map(|s| s.as_str()).unwrap_or("0");
                (
                    format!("{}.split({}, {})?", args[0], split_size, dim),
                    ReturnType::Vec,
                )
            }
            "torch.relu" | "relu" => (format!("{}.relu()?", args[0]), ReturnType::Tensor),
            "torch.sigmoid" | "sigmoid" => (
                format!("candle_nn::ops::sigmoid(&{})?", args[0]),
                ReturnType::Tensor,
            ),
            "torch.tanh" | "tanh" => (format!("{}.tanh()?", args[0]), ReturnType::Tensor),
            "torch.squeeze" | "squeeze" => {
                if args.len() > 1 {
                    (
                        format!("{}.squeeze({})?", args[0], args[1]),
                        ReturnType::Tensor,
                    )
                } else {
                    (format!("{}.squeeze(0)?", args[0]), ReturnType::Tensor)
                }
            }
            "torch.unsqueeze" | "unsqueeze" => (
                format!("{}.unsqueeze({})?", args[0], args[1]),
                ReturnType::Tensor,
            ),
            "torch.pow" | "pow" => (
                format!("{}.powf({})?", args[0], args[1]),
                ReturnType::Tensor,
            ),
            "torch.sqrt" | "sqrt" => (format!("{}.sqrt()?", args[0]), ReturnType::Tensor),
            "torch.exp" | "exp" => (format!("{}.exp()?", args[0]), ReturnType::Tensor),
            "torch.log" | "log" => (format!("{}.log()?", args[0]), ReturnType::Tensor),
            "torch.abs" | "abs" => (format!("{}.abs()?", args[0]), ReturnType::Tensor),
            "torch.sum" | "sum" => {
                if args.len() > 1 {
                    (format!("{}.sum({})?", args[0], args[1]), ReturnType::Tensor)
                } else {
                    (format!("{}.sum_all()?", args[0]), ReturnType::Tensor)
                }
            }
            "torch.mean" | "mean" => {
                if args.len() > 1 {
                    (
                        format!("{}.mean({})?", args[0], args[1]),
                        ReturnType::Tensor,
                    )
                } else {
                    (format!("{}.mean_all()?", args[0]), ReturnType::Tensor)
                }
            }
            "torch.transpose" => (
                format!("{}.transpose({}, {})?", args[0], args[1], args[2]),
                ReturnType::Tensor,
            ),
            "torch.reshape" => (
                format!("{}.reshape({})?", args[0], args[1]),
                ReturnType::Tensor,
            ),
            "torch.permute" => (
                format!("{}.permute({})?", args[0], args[1]),
                ReturnType::Tensor,
            ),
            // FEATURE: Comparisons
            "torch.lt" | "lt" | "_operator.lt" => {
                (format!("{}.lt(&{})?", args[0], args[1]), ReturnType::Tensor)
            }
            "torch.gt" | "gt" | "_operator.gt" => {
                (format!("{}.gt(&{})?", args[0], args[1]), ReturnType::Tensor)
            }
            // FEATURE: SDPA
            "torch.nn.functional.scaled_dot_product_attention"
            | "scaled_dot_product_attention"
            | "torch._C._nn.scaled_dot_product_attention" => {
                let q = &args[0];
                let k = &args[1];
                let v = &args[2];

                // Parse kwargs
                let attn_mask_arg = kwargs.get("attn_mask").and_then(|v| v.as_str());
                let attn_mask = if let Some(mask_name) = attn_mask_arg {
                    // mask_name is the variable name in Python. Look it up in var_map.
                    // If var_map has it, use it. If not, it might be "None".
                    if mask_name == "None" || mask_name.is_empty() {
                        "None".to_string()
                    } else {
                        format!(
                            "Some(&{})",
                            var_map.get(mask_name).unwrap_or(&mask_name.to_string())
                        )
                    }
                } else {
                    "None".to_string()
                };

                let dropout_p = kwargs
                    .get("dropout_p")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.0);

                let is_causal = kwargs
                    .get("is_causal")
                    .and_then(|v| v.as_bool())
                    .unwrap_or(false);

                // scale is sometimes a kwarg, sometimes optional positional.
                // For now assuming default scale in ops.rs if not present.
                let scale = kwargs
                    .get("scale")
                    .and_then(|v| v.as_f64())
                    .map(|f| format!("Some({})", f))
                    .unwrap_or("None".to_string());

                (
                    format!(
                        "pycandle_core::ops::scaled_dot_product_attention({}, {}, {}, {}, {:.1}, {}, {})?",
                        q, k, v, attn_mask, dropout_p, is_causal, scale
                    ),
                    ReturnType::Tensor,
                )
            }
            // FEATURE: Functional Tensor Creation
            "torch.ones" | "ones" => {
                let shape = args[0].clone();
                // Infer device from a previous tensor if available, else default?
                // We will try to find the first available tensor variable in scope to steal its device/dtype
                // or default to arbitrary choice if none (which might fail compile, but better than nothing)
                let device_hint = var_map
                    .values()
                    .next()
                    .map(|v| format!("{}.device()", v))
                    .unwrap_or("Device::Cpu".to_string());
                let dtype_hint = var_map
                    .values()
                    .next()
                    .map(|v| format!("{}.dtype()", v))
                    .unwrap_or("DType::F32".to_string());

                // If the graph has inputs, use the first one (usually 'xs')
                let (dev, dt) = if !var_map.is_empty() {
                    // Try to find a variable that is a Tensor
                    let tensor_var = var_map
                        .iter()
                        .find(|(k, _)| node_types.get(*k) == Some(&ReturnType::Tensor));
                    if let Some((_, v)) = tensor_var {
                        (format!("{}.device()", v), format!("{}.dtype()", v))
                    } else {
                        (device_hint, dtype_hint)
                    }
                } else {
                    ("Device::Cpu".to_string(), "DType::F32".to_string())
                };

                (
                    format!("Tensor::ones({}, {}, {})?", shape, dt, dev),
                    ReturnType::Tensor,
                )
            }
            "torch.zeros" | "zeros" => {
                let shape = args[0].clone();
                // Heuristic: Use first variable found for device/dtype
                let tensor_var = var_map
                    .iter()
                    .filter(|(k, _)| {
                        // Heuristic: Avoid binary op results which might be primitives masquerading as Tensors in incomplete type tracking
                        let k = k.as_str();
                        !k.starts_with("add")
                            && !k.starts_with("sub")
                            && !k.starts_with("mul")
                            && !k.starts_with("div")
                            && !k.starts_with("get")
                            && !k.starts_with("size")
                    })
                    .find(|(k, _)| node_types.get(*k) == Some(&ReturnType::Tensor));
                let (dev, dt) = if let Some((_, v)) = tensor_var {
                    (format!("{}.device()", v), format!("{}.dtype()", v))
                } else {
                    ("Device::Cpu".to_string(), "DType::F32".to_string())
                };
                (
                    format!("Tensor::zeros({}, {}, {})?", shape, dt, dev),
                    ReturnType::Tensor,
                )
            }
            "torch.arange" | "arange" => {
                // arange(start, end, step) or arange(end)
                // We need to check arg count
                let tensor_var = var_map
                    .iter()
                    .filter(|(k, _)| {
                        // Heuristic: Avoid binary op results which might be primitives masquerading as Tensors in incomplete type tracking
                        let k = k.as_str();
                        !k.starts_with("add")
                            && !k.starts_with("sub")
                            && !k.starts_with("mul")
                            && !k.starts_with("div")
                            && !k.starts_with("get")
                            && !k.starts_with("size")
                    })
                    .find(|(k, _)| node_types.get(*k) == Some(&ReturnType::Tensor));
                let dev = if let Some((_, v)) = tensor_var {
                    format!("{}.device()", v)
                } else {
                    "Device::Cpu".to_string()
                };

                if args.len() == 1 {
                    (
                        format!("Tensor::arange(0u32, {}, {})?", args[0], dev),
                        ReturnType::Tensor,
                    )
                } else if args.len() >= 2 {
                    (
                        format!("Tensor::arange({}, {}, {})?", args[0], args[1], dev),
                        ReturnType::Tensor,
                    )
                } else {
                    (
                        "Tensor::arange(0u32, 1u32, Device::Cpu)?".to_string(),
                        ReturnType::Tensor,
                    )
                }
            }
            "torch.full" => {
                let shape = args[0].clone();
                let fill_value = args[1].clone();
                let tensor_var = var_map
                    .iter()
                    .filter(|(k, _)| {
                        // Heuristic: Avoid binary op results which might be primitives masquerading as Tensors in incomplete type tracking
                        let k = k.as_str();
                        !k.starts_with("add")
                            && !k.starts_with("sub")
                            && !k.starts_with("mul")
                            && !k.starts_with("div")
                            && !k.starts_with("get")
                            && !k.starts_with("size")
                    })
                    .find(|(k, _)| node_types.get(*k) == Some(&ReturnType::Tensor));
                let (dev, dt) = if let Some((_, v)) = tensor_var {
                    (format!("{}.device()", v), format!("{}.dtype()", v))
                } else {
                    ("Device::Cpu".to_string(), "DType::F32".to_string())
                };
                (
                    format!(
                        "Tensor::full({}, {}, {})?.to_dtype({})?",
                        fill_value, shape, dev, dt
                    ),
                    ReturnType::Tensor,
                )
            }

            "operator.getitem" | "_operator.getitem" => {
                let idx = &args[1];

                // Find the source variable name to check its type
                let src_name = var_map
                    .iter()
                    .find(|(_, v)| *v == &args[0])
                    .map(|(k, _)| k.as_str())
                    .unwrap_or(&args[0]);

                let src_type = node_types
                    .get(src_name)
                    .cloned()
                    .unwrap_or(ReturnType::Tensor);

                match src_type {
                    ReturnType::Tuple => {
                        // Tuple indexing: x.0, x.1
                        if let Ok(i) = idx.parse::<usize>() {
                            (format!("{}.{}", args[0], i), ReturnType::Tensor)
                        } else {
                            (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor)
                        }
                    }
                    ReturnType::Vec => {
                        // Vec indexing: x[0]
                        if let Ok(i) = idx.parse::<usize>() {
                            (format!("{}[{}].clone()", args[0], i), ReturnType::Tensor)
                        } else {
                            (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor)
                        }
                    }
                    ReturnType::Tensor => {
                        if idx.contains("slice(") || idx == "None" || idx.starts_with("&[") {
                            // Map to .i() for indexing/slicing
                            let mut cleaned = idx.clone();
                            if cleaned.starts_with("&[") {
                                cleaned = cleaned[2..cleaned.len() - 1].to_string();
                            }

                            let items: Vec<String> = cleaned
                                .split(",")
                                .enumerate()
                                .map(|(i, s)| self.parse_slice_item(s.trim(), &args[0], i))
                                .collect();

                            let final_idx = if items.len() == 1 {
                                items[0].clone()
                            } else {
                                format!("({})", items.join(", "))
                            };

                            (format!("{}.i({})?", args[0], final_idx), ReturnType::Tensor)
                        } else {
                            (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor)
                        }
                    }
                    ReturnType::Primitive => (
                        format!("todo!(/* primitive indexing on {} */)", args[0]),
                        ReturnType::Primitive,
                    ),
                }
            }
            _ => {
                if target_lower.contains("add") && args.len() >= 2 {
                    return (
                        format!("(&{} + &{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }
                // FEATURE: In-Place Operations
                if target_lower.contains("add_") && args.len() >= 2 {
                    return (
                        format!("(&{} + &{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }

                // FEATURE: Non-Tensor Literals handling
                // We need to wrap numerical args in Tensor if they are being used in a tensor operation
                // This is a bit tricky without full type inference, but we can try to wrap obvious ones
                // or just rely on Candle's impl which often accepts f64 for some scalar ops.
                // However, things like `conv1d` expect tensors.
                // For now, let's just make sure we print unknown ops nicely.
                (
                    format!("todo!(/* function: {} */)", target),
                    ReturnType::Tensor,
                )
            }
        }
    }

    fn parse_slice_item(&self, item: &str, tensor_name: &str, dim_idx: usize) -> String {
        if item == "None" {
            return "..".to_string();
        }

        // Handle single negative index
        if let Ok(val) = item.parse::<isize>() {
            if val < 0 {
                return format!("{}.dim({})? - {}", tensor_name, dim_idx, val.abs());
            }
            return item.to_string();
        }

        if !item.contains("slice(") {
            return item.to_string();
        }

        // Parse slice(start, stop, step)
        let content = item
            .strip_prefix("slice(")
            .and_then(|s| s.strip_suffix(")"))
            .unwrap_or(item);
        let parts: Vec<&str> = content.split(",").map(|s| s.trim()).collect();

        let start_str = parts.get(0).copied().unwrap_or("None");
        let stop_str = parts.get(1).copied().unwrap_or("None");

        let start = if let Ok(val) = start_str.parse::<isize>() {
            if val < 0 {
                format!("{}.dim({})? - {}", tensor_name, dim_idx, val.abs())
            } else {
                start_str.to_string()
            }
        } else {
            start_str.to_string()
        };

        let stop = if let Ok(val) = stop_str.parse::<isize>() {
            if val < 0 {
                format!("{}.dim({})? - {}", tensor_name, dim_idx, val.abs())
            } else {
                stop_str.to_string()
            }
        } else {
            stop_str.to_string()
        };

        match (start.as_str(), stop.as_str()) {
            ("None", "None") => "..".to_string(),
            ("None", stop) => format!("..{}", stop),
            (start, "None") => format!("{}..", start),
            (start, stop) => format!("{}..{}", start, stop),
        }
    }

    fn map_fx_method(
        &self,
        method: &str,
        self_var: &str,
        self_var_name: &str,
        args: &[String],
        node_types: &HashMap<String, ReturnType>,
    ) -> (String, ReturnType) {
        let _src_type = node_types
            .get(self_var_name)
            .cloned()
            .unwrap_or(ReturnType::Tensor);

        match method {
            "view" | "reshape" => {
                if args.len() == 1 && args[0].starts_with("&[") {
                    (
                        format!("{}.reshape({})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                } else {
                    (
                        format!("{}.reshape(vec![{}])?", self_var, args.join(", ")),
                        ReturnType::Tensor,
                    )
                }
            }
            "flatten" => (format!("{}.flatten_all()?", self_var), ReturnType::Tensor),
            "transpose" => (
                format!("{}.transpose({}, {})?", self_var, args[0], args[1]),
                ReturnType::Tensor,
            ),
            "permute" => (
                format!("{}.permute({})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "unsqueeze" => (
                format!("{}.unsqueeze({})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "squeeze" => (
                format!("{}.squeeze({})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "chunk" => {
                let chunks = args.get(0).map(|s| s.as_str()).unwrap_or("2");
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("0");
                (
                    format!("{}.chunk({}, {})?", self_var, chunks, dim),
                    ReturnType::Vec,
                )
            }
            "split" => {
                let split_size = args.get(0).map(|s| s.as_str()).unwrap_or("1");
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                (
                    format!("{}.split({}, {})?", self_var, split_size, dim),
                    ReturnType::Vec,
                )
            }
            "t" => (format!("{}.t()?", self_var), ReturnType::Tensor),
            "contiguous" => (format!("{}.contiguous()?", self_var), ReturnType::Tensor),
            "size" => {
                if args.is_empty() {
                    (format!("{}.dims().to_vec()", self_var), ReturnType::Vec)
                } else {
                    (
                        format!("{}.dim({})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                }
            }
            // FEATURE: In-Place Operations (Method calls)
            "add_" => (
                format!("(&{} + &{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "sub_" => (
                format!("(&{} - &{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "mul_" => (
                format!("(&{} * &{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "div_" => (
                format!("(&{} / &{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            // Comparisons
            "lt" | "_operator.lt" => (
                format!("{}.lt(&{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            "gt" | "_operator.gt" => (
                format!("{}.gt(&{})?", self_var, args[0]),
                ReturnType::Tensor,
            ),
            // Ops
            "scaled_dot_product_attention" => (
                // Naive mapping to a potential ops::sdpa or just a placeholder if not existing
                // Since this is a method call on a module usually, but here it might be functional.
                // If functional: F.scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False)
                // We'll map to a custom helper or todo with clearer message if candle doesn't have it directly.
                // Candle has candle_nn::ops::softmax usually used.
                // Let's assume we map it to a custom function or just todo for now but CLEANLY.
                // actually candle-nn has it? No.
                // We'll emit a TODO but with the arguments.
                format!(
                    "todo!(\"scaled_dot_product_attention({}, {}, {}, ...)\")",
                    args.get(0).unwrap_or(&"".to_string()),
                    args.get(1).unwrap_or(&"".to_string()),
                    args.get(2).unwrap_or(&"".to_string())
                ),
                ReturnType::Tensor,
            ),
            "masked_fill_" => {
                // In-place masked_fill_: tensor.masked_fill_(mask, value)
                // Candle out-of-place: tensor.masked_fill(mask, value)
                let mask = &args[0];
                let value = &args[1];
                (
                    format!("{}.masked_fill(&{}, {})?", self_var, mask, value),
                    ReturnType::Tensor,
                )
            }
            _ => (
                format!("todo!(/* method: {} on {} */)", method, self_var),
                ReturnType::Tensor,
            ),
        }
    }

    fn infer_linear_dims(&self, meta: &LayerMeta) -> (usize, usize) {
        let in_f = meta.config.get("in_features").and_then(|v| v.as_u64());
        let out_f = meta.config.get("out_features").and_then(|v| v.as_u64());

        if let (Some(i), Some(o)) = (in_f, out_f) {
            return (i as usize, o as usize);
        }

        // Fallback to shapes if config is missing (common in ONNX or custom layers)
        let in_shape = meta
            .input_shapes
            .first()
            .and_then(|s| s.last())
            .copied()
            .unwrap_or(0);
        let out_shape = meta
            .output_shapes
            .first()
            .and_then(|s| s.last())
            .copied()
            .unwrap_or(0);

        (in_shape, out_shape)
    }

    fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
        // Check GPT2 helper first
        if let Some(init) = gpt2::generate_init(layer_name, meta, &self.config.dims) {
            return init;
        }

        // Core types
        // NEW: Check for weight norm flag
        let is_weight_norm = meta
            .config
            .get("weight_norm")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        match meta.module_type.as_str() {
            "Linear" | "LoRACompatibleLinear" | "Conv1D" => {
                let (in_f_val, out_f_val) = self.infer_linear_dims(meta);
                let in_f = self.render_dim(in_f_val, "hidden_dim");
                let out_f = self.render_dim(out_f_val, "");
                let bias = meta.config["bias"].as_bool().unwrap_or(true);

                if is_weight_norm {
                    // Generate call to the new helper
                    return format!(
                        "pycandle_core::layers::load_weight_norm_linear(vb.pp(\"{}\"), {}, {}, {})?",
                        layer_name, in_f, out_f, bias
                    );
                }

                // Check for weight shape to detect transpose needs
                let needs_transpose = meta
                    .config
                    .get("weight_shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let dims: Vec<u64> = arr.iter().filter_map(|x| x.as_u64()).collect();
                        // PyTorch Linear stores (out, in), if we see (in, out) we need transpose
                        dims.len() == 2 && dims[0] == in_f_val as u64 && dims[1] == out_f_val as u64
                    })
                    .unwrap_or(false);

                if needs_transpose {
                    format!(
                        "{{ let w = vb.pp(\"{}\").get(({}, {}), \"weight\")?.t()?; \
                         let b = {}; candle_nn::Linear::new(w, b) }}",
                        layer_name,
                        in_f,
                        out_f,
                        if bias {
                            format!("Some(vb.pp(\"{}\").get({}, \"bias\")?)", layer_name, out_f)
                        } else {
                            "None".to_string()
                        }
                    )
                } else if bias {
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
                let in_c = self.render_dim(
                    meta.config["in_channels"].as_u64().unwrap_or(0) as usize,
                    "",
                );
                let out_c = self.render_dim(
                    meta.config["out_channels"].as_u64().unwrap_or(0) as usize,
                    "",
                );
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
                // Use a single value if shape is [N], otherwise use a Slice
                let shape_str = if shape.len() == 1 {
                    format!("{}", shape[0])
                } else {
                    format!("Shape::from(vec!{:?})", shape)
                };
                format!(
                    "candle_nn::layer_norm({}, candle_nn::LayerNormConfig {{ eps: {:.1e}, ..Default::default() }}, vb.pp(\"{}\"))?",
                    shape_str, eps, layer_name
                )
            }
            "Embedding" => {
                let n = self.render_dim(
                    meta.config["num_embeddings"].as_u64().unwrap_or(0) as usize,
                    "vocab_size",
                );
                let d = self.render_dim(
                    meta.config["embedding_dim"].as_u64().unwrap_or(0) as usize,
                    "hidden_dim",
                );
                format!(
                    "candle_nn::embedding({}, {}, vb.pp(\"{}\"))?",
                    n, d, layer_name
                )
            }
            // Activations - stateless
            "ReLU" => "ReLU".to_string(),
            "GELU" => "GELU".to_string(),
            "Sigmoid" => "Sigmoid".to_string(),
            "Tanh" => "Tanh".to_string(),
            // Activations - parameterized
            "ELU" => {
                let alpha = meta
                    .config
                    .get("alpha")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(1.0);
                format!("ELU::new({})", alpha)
            }
            "LeakyReLU" => {
                let slope = meta
                    .config
                    .get("negative_slope")
                    .and_then(|v| v.as_f64())
                    .unwrap_or(0.01);
                format!("LeakyReLU::new({})", slope)
            }
            "Snake" => {
                let in_features = meta
                    .config
                    .get("in_features")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                format!("Snake::load(vb.pp(\"{}\"), {})?", layer_name, in_features)
            }
            "BatchNorm1d" => {
                let num_features = meta
                    .config
                    .get("num_features")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                format!(
                    "BatchNorm1d::load(vb.pp(\"{}\"), {})?",
                    layer_name, num_features
                )
            }
            "BatchNorm2d" => {
                let num_features = meta
                    .config
                    .get("num_features")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                format!(
                    "BatchNorm2d::load(vb.pp(\"{}\"), {})?",
                    layer_name, num_features
                )
            }
            "LSTM" => {
                let input_size = self.render_dim(
                    meta.config
                        .get("input_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    "hidden_dim",
                );
                let hidden_size = self.render_dim(
                    meta.config
                        .get("hidden_size")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(0) as usize,
                    "",
                );
                let num_layers = self.render_dim(
                    meta.config
                        .get("num_layers")
                        .and_then(|v| v.as_u64())
                        .unwrap_or(1) as usize,
                    "n_layers",
                );
                format!(
                    "LSTM::load(vb.pp(\"{}\"), {}, {}, {})?",
                    layer_name, input_size, hidden_size, num_layers
                )
            }
            "CausalConv1d" => {
                let in_c = self.render_dim(
                    meta.config["in_channels"].as_u64().unwrap_or(0) as usize,
                    "",
                );
                let out_c = self.render_dim(
                    meta.config["out_channels"].as_u64().unwrap_or(0) as usize,
                    "",
                );
                let k = meta.config["kernel_size"].as_u64().unwrap_or(0);
                let s = meta.config["stride"].as_u64().unwrap_or(1);
                let bias = meta.config["bias"].as_bool().unwrap_or(true);
                format!(
                    "CausalConv1d::load(vb.pp(\"{}\"), {}, {}, {}, {}, {})?",
                    layer_name, in_c, out_c, k, s, bias
                )
            }
            "Mish" => "Mish".to_string(),
            "SiLU" => "SiLU".to_string(),
            // GPT2 specific
            "NewGELUActivation" => "candle_nn::Activation::NewGelu".to_string(),
            "Dropout" => "Dropout::new()".to_string(),
            "Transpose" => {
                let d0 = meta.config["dim0"].as_u64().unwrap_or(1);
                let d1 = meta.config["dim1"].as_u64().unwrap_or(2);
                format!("Transpose::new({}, {})", d0, d1)
            }
            "SinusoidalPosEmb" => {
                let dim = meta.output_shapes[0][1];
                format!("SinusoidalPosEmb::new({})", dim)
            }
            _ => format!(
                "todo!(\"Implement initialization for {}\")",
                meta.module_type
            ),
        }
    }

    fn get_forward_return_type(&self) -> String {
        if !self.graph_nodes.is_empty() {
            for node in &self.graph_nodes {
                if node.op == "output" {
                    if let Some(arg0) = node.args.get(0) {
                        if let Some(arr) = arg0.as_array() {
                            if arr.len() > 1 {
                                let tensors = vec!["Tensor"; arr.len()];
                                return format!("Result<({})>", tensors.join(", "));
                            }
                        }
                    }
                }
            }
        }
        "Result<Tensor>".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_sanitize_name() {
        let codegen = Codegen::new(HashMap::new(), None);

        // PyTorch names
        assert_eq!(
            codegen.sanitize_name("encoder.layers.0"),
            "encoder_layers_0"
        );
        assert_eq!(codegen.sanitize_name("node_123"), "x_123");
        assert_eq!(codegen.sanitize_name("123_invalid"), "x_123_invalid");

        // ONNX names
        assert_eq!(codegen.sanitize_name("/layers/0/Gemm_output"), "gemm");
        assert_eq!(codegen.sanitize_name("/node_456"), "x_456");
        assert_eq!(codegen.sanitize_name("/Gather_1_output"), "gather_1");
    }
}
