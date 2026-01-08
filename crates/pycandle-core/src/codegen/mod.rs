//! Code generation from PyTorch manifests to Candle Rust code
//!
//! This module generates idiomatic Rust code from recorded PyTorch model manifests.

pub mod gpt2;

use crate::LayerMeta;
use serde::Serialize;
use std::collections::HashMap;

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

/// Code generator that converts manifests to Rust code
pub struct Codegen {
    manifest: HashMap<String, LayerMeta>,
    graph_nodes: Option<Vec<GraphNode>>,
}

impl Codegen {
    pub fn new(manifest: HashMap<String, LayerMeta>) -> Self {
        Self {
            manifest,
            graph_nodes: None,
        }
    }

    pub fn with_graph(mut self, graph_nodes: Vec<GraphNode>) -> Self {
        self.graph_nodes = Some(graph_nodes);
        self
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

        if let Some(nodes) = &self.graph_nodes {
            code.push_str(&self.generate_forward_dag(nodes));
        } else {
            // Forward method
            code.push_str("    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {\n");
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
        code.push_str("    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {\n");

        // Map python placeholder names to Rust input variables
        let mut var_map = HashMap::new();

        for node in nodes {
            match node.op.as_str() {
                "placeholder" => {
                    // Assume first placeholder is xs
                    var_map.insert(node.name.clone(), "xs".to_string());
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

                    let forward_call =
                        if self.map_type(&node.module_type.clone().unwrap_or_default()) == "LSTM" {
                            format!("self.{}.forward(&{})?.0", clean_name, input_var)
                        } else {
                            format!("self.{}.forward(&{})?", clean_name, input_var)
                        };

                    code.push_str(&format!("        let {} = {};\n", var_name, forward_call));
                    code.push_str(&format!(
                        "        py_check!(self.checker, \"{}\", &{});\n",
                        node.target, var_name
                    ));
                    var_map.insert(node.name.clone(), var_name);
                }
                "call_function" => {
                    let var_name = format!("x_{}", node.name);
                    let mut resolved_args = Vec::new();
                    for arg in &node.args {
                        resolved_args.push(self.resolve_fx_arg(arg, &var_map));
                    }

                    let expr = self.map_fx_op(&node.target, &resolved_args);
                    code.push_str(&format!("        let {} = {};\n", var_name, expr));
                    var_map.insert(node.name.clone(), var_name);
                }
                "call_method" => {
                    let var_name = format!("x_{}", node.name);
                    let mut resolved_args = Vec::new();
                    for arg in &node.args {
                        resolved_args.push(self.resolve_fx_arg(arg, &var_map));
                    }

                    if !resolved_args.is_empty() {
                        let self_var = &resolved_args[0];
                        let method_args = &resolved_args[1..];
                        let expr = self.map_fx_method(&node.target, self_var, method_args);
                        code.push_str(&format!("        let {} = {};\n", var_name, expr));
                        var_map.insert(node.name.clone(), var_name);
                    }
                }
                "output" => {
                    let out_var = if let Some(arg0) = node.args.get(0) {
                        match arg0 {
                            serde_json::Value::String(s) => {
                                var_map.get(s).cloned().unwrap_or(s.clone())
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

    fn map_fx_op(&self, target: &str, args: &[String]) -> String {
        let target_lower = target.to_lowercase();
        if target_lower.contains("add") {
            return format!("(&{} + &{})?", args[0], args[1]);
        }
        if target_lower.contains("sub") {
            return format!("(&{} - &{})?", args[0], args[1]);
        }
        if target_lower.contains("mul") {
            return format!("(&{} * &{})?", args[0], args[1]);
        }
        if target_lower.contains("div") || target_lower.contains("truediv") {
            return format!("(&{} / &{})?", args[0], args[1]);
        }
        if target_lower.contains("cat") {
            let dim = args.get(1).map(|s| s.as_str()).unwrap_or("1");
            return format!("Tensor::cat({}, {})?", args[0], dim);
        }

        match target {
            "torch.add" | "<built-in function add>" | "add" => {
                format!("(&{} + &{})?", args[0], args[1])
            }
            "torch.cat" | "cat" => {
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                format!("Tensor::cat({}, {})?", args[0], dim)
            }
            "torch.relu" | "relu" => format!("{}.relu()?", args[0]),
            "torch.sigmoid" | "sigmoid" => format!("candle_nn::ops::sigmoid(&{})?", args[0]),
            "torch.tanh" | "tanh" => format!("{}.tanh()?", args[0]),
            "operator.getitem" => {
                // Often used for slicing or getting a member.
                // args[0] is the tensor, args[1] is the index
                format!("{}.get({})?", args[0], args[1])
            }
            _ => format!("todo!(/* function: {} */)", target),
        }
    }

    fn map_fx_method(&self, method: &str, self_var: &str, args: &[String]) -> String {
        match method {
            "view" | "reshape" => {
                // PyTorch view often has (-1) or (B, -1).
                // We need to map it to Candle reshape.
                // args might be (shape_list)
                if args.len() == 1 && args[0].starts_with("&[") {
                    format!("{}.reshape({})?", self_var, args[0])
                } else {
                    format!("{}.reshape(vec![{}])?", self_var, args.join(", "))
                }
            }
            "flatten" => {
                format!("{}.flatten_all()?", self_var)
            }
            "transpose" => {
                format!("{}.transpose({}, {})?", self_var, args[0], args[1])
            }
            "permute" => {
                format!("{}.permute({})?", self_var, args[0])
            }
            "t" => {
                format!("{}.t()?", self_var)
            }
            "contiguous" => {
                format!("{}.contiguous()?", self_var)
            }
            "size" => {
                if args.is_empty() {
                    format!("{}.dims().to_vec()", self_var)
                } else {
                    format!("{}.dim({})?", self_var, args[0])
                }
            }
            _ => format!("todo!(/* method: {} on {} */)", method, self_var),
        }
    }

    fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
        // Check GPT2 helper first
        if let Some(init) = gpt2::generate_init(layer_name, meta) {
            return init;
        }

        // Core types
        match meta.module_type.as_str() {
            "Linear" | "LoRACompatibleLinear" => {
                let in_f = meta.config["in_features"].as_u64().unwrap_or(0);
                let out_f = meta.config["out_features"].as_u64().unwrap_or(0);
                let bias = meta.config["bias"].as_bool().unwrap_or(true);

                // Check for weight shape to detect transpose needs
                let needs_transpose = meta
                    .config
                    .get("weight_shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let dims: Vec<u64> = arr.iter().filter_map(|x| x.as_u64()).collect();
                        // PyTorch Linear stores (out, in), if we see (in, out) we need transpose
                        dims.len() == 2 && dims[0] == in_f && dims[1] == out_f
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
                let n = meta.config["num_embeddings"].as_u64().unwrap_or(0);
                let d = meta.config["embedding_dim"].as_u64().unwrap_or(0);
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
                let input_size = meta
                    .config
                    .get("input_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let hidden_size = meta
                    .config
                    .get("hidden_size")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0) as usize;
                let num_layers = meta
                    .config
                    .get("num_layers")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(1) as usize;
                format!(
                    "LSTM::load(vb.pp(\"{}\"), {}, {}, {})?",
                    layer_name, input_size, hidden_size, num_layers
                )
            }
            "CausalConv1d" => {
                let in_c = meta.config["in_channels"].as_u64().unwrap_or(0);
                let out_c = meta.config["out_channels"].as_u64().unwrap_or(0);
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
                let out_f = meta.config["nf"].as_u64().unwrap_or(0);
                // In GPT2, weights are (nx, nf), so no transpose needed relative to standard HF Conv1D logic
                // But Candle Linear wants (out, in).
                // "weight_shape" should guide us.
                // Assuming we use standard Linear and handle weights in loading.
                let nx_guess =
                    if let Some(ws) = meta.config.get("weight_shape").and_then(|v| v.as_array()) {
                        // (nx, nf)
                        ws.get(0).and_then(|x| x.as_u64()).unwrap_or(0)
                    } else {
                        0
                    };

                format!(
                    "candle_nn::linear({}, {}, vb.pp(\"{}\"))?",
                    nx_guess, out_f, layer_name
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
}
