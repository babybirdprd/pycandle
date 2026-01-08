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
    pub module_type: Option<String>,
}

#[derive(Serialize, serde::Deserialize, Debug, Clone, Default)]
pub struct SymbolicConfig {
    pub dims: HashMap<String, usize>,
}

/// Code generator that converts manifests to Rust code
pub struct Codegen {
    manifest: HashMap<String, LayerMeta>,
    graph_nodes: Option<Vec<GraphNode>>,
    config: SymbolicConfig,
    hints: HashMap<String, usize>,
}

impl Codegen {
    pub fn new(
        manifest: HashMap<String, LayerMeta>,
        hints: Option<HashMap<String, usize>>,
    ) -> Self {
        let mut slf = Self {
            manifest,
            graph_nodes: None,
            config: SymbolicConfig::default(),
            hints: hints.unwrap_or_default(),
        };
        slf.config = slf.extract_symbolic_config();
        slf
    }

    pub fn with_graph(mut self, graph_nodes: Vec<GraphNode>) -> Self {
        self.graph_nodes = Some(graph_nodes);
        self
    }

    pub fn extract_symbolic_config(&self) -> SymbolicConfig {
        let mut dims = self.hints.clone();

        for meta in self.manifest.values() {
            // GPT2 specific extraction
            if let Some(v) = meta.config.get("vocab_size").and_then(|v| v.as_u64()) {
                dims.insert("vocab_size".to_string(), v as usize);
            }
            if let Some(v) = meta.config.get("n_embd").and_then(|v| v.as_u64()) {
                dims.insert("hidden_dim".to_string(), v as usize);
            }
            if let Some(v) = meta.config.get("n_head").and_then(|v| v.as_u64()) {
                dims.insert("n_head".to_string(), v as usize);
            }
            if let Some(v) = meta.config.get("n_layer").and_then(|v| v.as_u64()) {
                dims.insert("n_layers".to_string(), v as usize);
            }
            if let Some(v) = meta.config.get("n_positions").and_then(|v| v.as_u64()) {
                dims.insert("context_length".to_string(), v as usize);
            }

            // Generic extraction
            match meta.module_type.as_str() {
                "Embedding" => {
                    if let Some(n) = meta.config.get("num_embeddings").and_then(|v| v.as_u64()) {
                        dims.insert("vocab_size".to_string(), n as usize);
                    }
                    if let Some(d) = meta.config.get("embedding_dim").and_then(|v| v.as_u64()) {
                        dims.insert("hidden_dim".to_string(), d as usize);
                    }
                }
                "Linear" | "LoRACompatibleLinear" => {
                    // If out_features is high (like 50000+), maybe it's vocab_size?
                    if let Some(out_f) = meta.config.get("out_features").and_then(|v| v.as_u64()) {
                        if out_f > 30000 && !dims.contains_key("vocab_size") {
                            dims.insert("vocab_size".to_string(), out_f as usize);
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
                    return format!("cfg.{}", preferred_name);
                }
            }
        }

        // Try exact value match in any dim
        for (name, &v) in &self.config.dims {
            if v == value {
                return format!("cfg.{}", name);
            }
        }

        value.to_string()
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
        code.push_str("use candle_core::{Tensor, Result, Device, Shape};\n");
        code.push_str(
            "use candle_nn::{Linear, Conv1d, LayerNorm, Embedding, VarBuilder, Module};\n",
        );
        code.push_str("use pycandle_core::{PyChecker, py_check, Dropout, Transpose, Mish, CausalConv1d, SiLU, ReLU, GELU, Sigmoid, Tanh, ELU, LeakyReLU, Snake, BatchNorm1d, BatchNorm2d, LSTM};\n");

        // Add gpt2 import if GPT2 types are present
        if self.has_gpt2_types() {
            code.push_str("use pycandle_core::gpt2;\n");
        }
        code.push_str("\n");

        if !self.config.dims.is_empty() {
            code.push_str(&self.generate_config_struct());
            code.push_str("\n\n");
        }

        code.push_str(&self.generate_struct(model_name));
        code.push_str("\n\n");
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
            "Conv1D" => "Linear".to_string(),
            "Dropout" => "Dropout".to_string(),
            "NewGELUActivation" => "candle_nn::Activation".to_string(),
            "LoRACompatibleLinear" => "Linear".to_string(),
            "SinusoidalPosEmb" => "SinusoidalPosEmb".to_string(),
            _ => format!("() /* TODO: {} */", py_type),
        }
    }

    fn generate_impl(&self, model_name: &str) -> String {
        let mut code = format!("impl {} {{\n", model_name);

        let load_args = if self.config.dims.is_empty() {
            "vb: VarBuilder, checker: Option<PyChecker>".to_string()
        } else {
            "cfg: Config, vb: VarBuilder, checker: Option<PyChecker>".to_string()
        };

        code.push_str(&format!(
            "    pub fn load({}) -> Result<Self> {{\n",
            load_args
        ));

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

        if let Some(nodes) = &self.graph_nodes {
            code.push_str(&self.generate_forward_dag(nodes));
        } else {
            // Forward method
            let ret_type = self.get_forward_return_type();
            code.push_str(&format!(
                "    pub fn forward(&self, xs: &Tensor) -> {} {{\n",
                ret_type
            ));
            code.push_str("        let mut x = xs.clone();\n");

            for (layer_name, meta) in &layers {
                if !meta.is_leaf {
                    continue;
                }
                let clean_name = layer_name.replace(".", "_");
                let forward_call = if self.map_type(&meta.module_type) == "LSTM" {
                    format!("self.{}.forward(&x)?.0", clean_name)
                } else {
                    format!("self.{}.forward(&x)?", clean_name)
                };
                code.push_str(&format!("\n        // Layer: {}\n", layer_name));
                code.push_str(&format!("        x = {};\n", forward_call));
                code.push_str(&format!(
                    "        py_check!(self.checker, \"{}\", &x);\n",
                    layer_name
                ));
            }

            code.push_str("\n        Ok(x)\n");
            code.push_str("    }\n");
        }
        code.push_str("}\n");

        code
    }

    fn generate_forward_dag(&self, nodes: &[GraphNode]) -> String {
        let mut code = String::new();
        let ret_type = self.get_forward_return_type();
        code.push_str(&format!(
            "    pub fn forward(&self, xs: &Tensor) -> {} {{\n",
            ret_type
        ));

        // Map python placeholder names to Rust input variables
        let mut var_map = HashMap::new();
        let mut node_types = HashMap::new();

        for node in nodes {
            match node.op.as_str() {
                "placeholder" => {
                    // Assume first placeholder is xs
                    var_map.insert(node.name.clone(), "xs".to_string());
                    node_types.insert(node.name.clone(), ReturnType::Tensor);
                }
                "call_module" => {
                    let clean_name = node.target.replace(".", "_");
                    let var_name = format!("x_{}", node.name);
                    let input_var = if let Some(arg0) = node.args.get(0) {
                        match arg0 {
                            serde_json::Value::String(s) => {
                                var_map.get(s).cloned().unwrap_or(s.clone())
                            }
                            _ => "xs".to_string(),
                        }
                    } else {
                        "xs".to_string()
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
                    let var_name = format!("x_{}", node.name);
                    let mut resolved_args = Vec::new();
                    for arg in &node.args {
                        resolved_args.push(self.resolve_fx_arg(arg, &var_map));
                    }

                    let (expr, return_type) =
                        self.map_fx_op(&node.target, &resolved_args, &var_map, &node_types);
                    code.push_str(&format!("        let {} = {};\n", var_name, expr));
                    var_map.insert(node.name.clone(), var_name);
                    node_types.insert(node.name.clone(), return_type);
                }
                "call_method" => {
                    let var_name = format!("x_{}", node.name);
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
                            _ => "xs".to_string(),
                        }
                    } else {
                        "xs".to_string()
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

        // Functional ops
        match target {
            "torch.cat" | "cat" => {
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                (
                    format!("Tensor::cat({}, {})?", args[0], dim),
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
            "operator.getitem" => {
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
                            cleaned = cleaned.replace("slice(None, None, None)", "..");
                            // Handle slice(None, stop, None) -> ..stop
                            while let Some(pos) = cleaned.find("slice(None, ") {
                                if let Some(end_pos) = cleaned[pos..].find(", None)") {
                                    let stop = &cleaned[pos + 12..pos + end_pos];
                                    let stop_trimmed = stop.trim();
                                    cleaned.replace_range(
                                        pos..pos + end_pos + 7,
                                        &format!("..{}", stop_trimmed),
                                    );
                                } else {
                                    break;
                                }
                            }
                            // Handle slice(start, None, None) -> start..
                            while let Some(pos) = cleaned.find("slice(") {
                                if let Some(end_pos) = cleaned[pos..].find(", None, None)") {
                                    let start = &cleaned[pos + 6..pos + end_pos];
                                    let start_trimmed = start.trim();
                                    cleaned.replace_range(
                                        pos..pos + end_pos + 13,
                                        &format!("{}..", start_trimmed),
                                    );
                                } else {
                                    break;
                                }
                            }
                            cleaned = cleaned.replace("None", "..");
                            (format!("{}.i(({}))?", args[0], cleaned), ReturnType::Tensor)
                        } else {
                            (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor)
                        }
                    }
                }
            }
            _ => {
                if target_lower.contains("add") && args.len() >= 2 {
                    return (
                        format!("(&{} + &{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }
                (
                    format!("todo!(/* function: {} */)", target),
                    ReturnType::Tensor,
                )
            }
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
        let src_type = node_types
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
                    (format!("{}.dims().to_vec()", self_var), ReturnType::Tensor)
                } else {
                    (
                        format!("{}.dim({})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                }
            }
            _ => (
                format!("todo!(/* method: {} on {} */)", method, self_var),
                ReturnType::Tensor,
            ),
        }
    }

    fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
        // Check GPT2 helper first
        if let Some(init) = gpt2::generate_init(layer_name, meta, &self.config.dims) {
            return init;
        }

        // Core types
        match meta.module_type.as_str() {
            "Linear" | "LoRACompatibleLinear" => {
                let in_f_val = meta.config["in_features"].as_u64().unwrap_or(0) as usize;
                let out_f_val = meta.config["out_features"].as_u64().unwrap_or(0) as usize;
                let in_f = self.render_dim(in_f_val, "hidden_dim");
                let out_f = self.render_dim(out_f_val, "");
                let bias = meta.config["bias"].as_bool().unwrap_or(true);

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
                         let b = {}; Linear::new(w, b) }}",
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
            "Conv1D" => {
                let out_f_val = meta.config["nf"].as_u64().unwrap_or(0) as usize;
                let out_f = self.render_dim(out_f_val, "");
                let nx_val =
                    if let Some(ws) = meta.config.get("weight_shape").and_then(|v| v.as_array()) {
                        ws.get(0).and_then(|x| x.as_u64()).unwrap_or(0) as usize
                    } else {
                        0
                    };
                let nx = self.render_dim(nx_val, "hidden_dim");

                format!(
                    "candle_nn::linear({}, {}, vb.pp(\"{}\"))?",
                    nx, out_f, layer_name
                )
            }
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
        if let Some(nodes) = &self.graph_nodes {
            for node in nodes {
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
