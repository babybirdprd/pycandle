use super::gpt2;
use crate::LayerMeta;
use regex::Regex;
use std::collections::{HashMap, HashSet};

struct GroupInfo {
    _name: String,
    count: usize,
    module_type: String,
}

impl super::Codegen {
    fn get_groups(&self) -> HashMap<String, GroupInfo> {
        let mut groups: HashMap<String, Vec<usize>> = HashMap::new();
        let mut types: HashMap<String, String> = HashMap::new();
        let re = Regex::new(r"^(.*)\.(\d+)$").unwrap();

        for (name, meta) in &self.manifest {
            if let Some(caps) = re.captures(name) {
                let base = caps.get(1).unwrap().as_str().to_string();
                let idx = caps.get(2).unwrap().as_str().parse::<usize>().unwrap();
                groups.entry(base.clone()).or_default().push(idx);
                types.insert(base, meta.module_type.clone());
            }
        }

        let mut result = HashMap::new();
        for (base, indices) in groups {
            // Only group if we have at least 2 items and they are roughly sequential
            if indices.len() > 1 {
                let max_idx = *indices.iter().max().unwrap();
                if max_idx == indices.len() - 1 {
                    result.insert(
                        base.clone(),
                        GroupInfo {
                            _name: base.clone(),
                            count: indices.len(),
                            module_type: types[&base].clone(),
                        },
                    );
                }
            }
        }
        result
    }

    fn resolve_target(&self, target: &str, groups: &HashMap<String, GroupInfo>) -> String {
        let re = Regex::new(r"^(.*)\.(\d+)$").unwrap();
        if let Some(caps) = re.captures(target) {
            let base = caps.get(1).unwrap().as_str();
            let idx = caps.get(2).unwrap().as_str();
            if groups.contains_key(base) {
                return format!("{}[{}]", self.sanitize_name(base), idx);
            }
        }
        self.sanitize_name(target)
    }

    pub fn generate_model_rs(&self, model_name: &str) -> String {
        let mut code = String::new();
        code.push_str("use candle_core::{Result, Tensor, IndexOp, Shape, Device, DType};\n");
        code.push_str("use candle_nn::{VarBuilder, Module};\n");
        code.push_str("use pycandle_core::layers::*;\n");
        code.push_str("use pycandle_core::weights;\n\n");

        // Generate Config struct
        code.push_str(&self.generate_config_struct());
        code.push_str("\n");

        if self.stateful {
            code.push_str(
                "#[derive(Debug, Clone)]
pub struct Cache {
    pub use_cache: bool,
    pub kv: Vec<Option<(Tensor, Tensor)>>,
    pub offset: usize,
}

impl Cache {
    pub fn new(use_cache: bool) -> Self {
       Self { use_cache, kv: Vec::new(), offset: 0 }
    }
}
\n",
            );
        }

        // Generate Model struct
        code.push_str(&self.generate_struct(model_name));
        code.push_str("\n");

        // Generate Implementation
        code.push_str(&self.generate_impl(model_name));

        code
    }

    fn generate_config_struct(&self) -> String {
        let mut code = String::new();
        code.push_str("#[derive(Debug, Clone)]\n");
        code.push_str("pub struct Config {\n");
        for name in self.config.dims.keys() {
            code.push_str(&format!("    pub {}: usize,\n", name));
        }
        code.push_str("}\n\n");

        code.push_str("impl Default for Config {\n");
        code.push_str("    fn default() -> Self {\n");
        code.push_str("        Self {\n");
        for (name, val) in &self.config.dims {
            code.push_str(&format!("            {}: {},\n", name, val));
        }
        code.push_str("        }\n");
        code.push_str("    }\n");
        code.push_str("}\n");
        code
    }

    fn generate_struct(&self, model_name: &str) -> String {
        let mut code = String::new();
        code.push_str(&format!("pub struct {} {{\n", model_name));

        let groups = self.get_groups();
        let mut handled_groups = HashSet::new();

        // Sort names for deterministic output
        let mut names: Vec<_> = self.manifest.keys().collect();
        names.sort();

        for name in names {
            // Check if part of a group
            let re = Regex::new(r"^(.*)\.(\d+)$").unwrap();
            if let Some(caps) = re.captures(name) {
                let base = caps.get(1).unwrap().as_str();
                if let Some(info) = groups.get(base) {
                    if handled_groups.contains(base) {
                        continue;
                    }
                    // Generate Vector field
                    let rust_type = self.map_type(&info.module_type);
                    code.push_str(&format!("    /// Group: {} (x{})\n", base, info.count));
                    code.push_str(&format!(
                        "    pub {}: Vec<{}>,\n",
                        self.sanitize_name(base),
                        rust_type
                    ));
                    handled_groups.insert(base.to_string());
                    continue;
                }
            }

            let meta = &self.manifest[name];
            let sanitized = self.sanitize_name(name);
            let rust_type = self.map_type(&meta.module_type);

            // Docstring Injection
            code.push_str(&format!("    /// Layer: {}\n", name));
            if !meta.input_shapes.is_empty() {
                code.push_str(&format!("    /// Input: {:?}\n", meta.input_shapes));
            }
            if !meta.parameters.is_empty() {
                code.push_str(&format!(
                    "    /// Params: {} tensors\n",
                    meta.parameters.len()
                ));
            }

            code.push_str(&format!("    pub {}: {},\n", sanitized, rust_type));
        }

        // Add config
        code.push_str("    pub config: Config,\n");

        code.push_str("}\n");
        code
    }

    fn generate_impl(&self, model_name: &str) -> String {
        let mut code = String::new();
        code.push_str(&format!("impl {} {{\n", model_name));

        // Load method
        code.push_str("    pub fn load_from_hub(repo: &str, revision: &str, device: &Device, config: Config) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo.to_string()).with_revision(revision.to_string());
        let path = repo.get(\"model.safetensors\")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], device.clone())? };
        Self::load(vb, config)
    }

    pub fn load(vb: VarBuilder, config: Config) -> Result<Self> {\n");
        code.push_str("        Ok(Self {\n");

        let groups = self.get_groups();
        let mut handled_groups = HashSet::new();
        let mut names: Vec<_> = self.manifest.keys().collect();
        names.sort();

        for name in names {
            let re = Regex::new(r"^(.*)\.(\d+)$").unwrap();
            if let Some(caps) = re.captures(name) {
                let base = caps.get(1).unwrap().as_str();
                if let Some(info) = groups.get(base) {
                    if handled_groups.contains(base) {
                        continue;
                    }
                    let sanitized = self.sanitize_name(base);
                    // Vector Load Loop
                    code.push_str(&format!(
                        "            {}: (0..{}).map(|i| {{\n",
                        sanitized, info.count
                    ));
                    // We need to generate init for one item to see the pattern, but with dynamic name
                    // Hack: Generate init for base.0, then replace base.0 with key using regex or format
                    // Better: The `generate_init` takes `layer_name`.
                    // We can call `generate_init` with `format!(\"{}.{}\", base, i)`?
                    // But `generate_init` looks up manifest. manifest needs real key.
                    // We have real keys in manifest.
                    // So inside the map loop, we can't easily call generate_init dynamic text unless we inline the logic.
                    // Or we assume all items in group are identical types (they are) and use base.0 as template?
                    // Yes, use base.0 as template, but replace "base.0" string in generated code with dynamic format.

                    let template_key = format!("{}.0", base);
                    let meta = &self.manifest[&template_key];
                    let init_code = self.generate_init(&template_key, meta);

                    // Replace "base.0" with "{}.{}", base, i in the logic?
                    // generate_init generates `vb.pp("base.0")`.
                    // We want `vb.pp(&format!("{}.{}", base, i))`.
                    let fixed_init = init_code.replace(
                        &format!("\"{}\"", template_key),
                        &format!("&format!(\"{}.{{}}\", i)", base),
                    );

                    code.push_str(&format!("                {}\n", fixed_init));
                    code.push_str("            }).collect::<Result<Vec<_>>>()?,\n");

                    handled_groups.insert(base.to_string());
                    continue;
                }
            }

            let meta = &self.manifest[name];
            let sanitized = self.sanitize_name(name);
            let init = self.generate_init(name, meta);
            code.push_str(&format!("            {}: {},\n", sanitized, init));
        }
        code.push_str("            config,\n");
        code.push_str("        })\n");
        code.push_str("    }\n\n");

        // Forward method
        let ret_type = self.get_forward_return_type();
        let forward_args = if self.stateful {
            "&self, xs: &Tensor, cache: &mut Cache"
        } else {
            "&self, xs: &Tensor"
        };
        code.push_str(&format!(
            "    pub fn forward({}) -> {} {{\n",
            forward_args, ret_type
        ));

        if !self.graph_nodes.is_empty() {
            code.push_str(&self.generate_forward_dag(&self.graph_nodes));
        } else {
            code.push_str("        let mut x = xs.clone();\n");

            // Sequential Strategy with Groups
            let groups = self.get_groups(); // Re-compute or reuse? 
            let mut handled_groups = HashSet::new();
            let mut names: Vec<_> = self.manifest.keys().collect();
            names.sort();

            for name in names {
                let re = Regex::new(r"^(.*)\.(\d+)$").unwrap();
                if let Some(caps) = re.captures(name) {
                    let base = caps.get(1).unwrap().as_str();
                    if let Some(_) = groups.get(base) {
                        if handled_groups.contains(base) {
                            continue;
                        }
                        let sanitized = self.sanitize_name(base);
                        code.push_str(&format!("        for layer in &self.{} {{\n", sanitized));
                        code.push_str("            x = layer.forward(&x)?;\n");
                        code.push_str("        }\n");
                        handled_groups.insert(base.to_string());
                        continue;
                    }
                }

                let sanitized = self.sanitize_name(name);
                code.push_str(&format!("        x = self.{}.forward(&x)?;\n", sanitized));
            }
            code.push_str("        Ok(x)\n");
        }

        code.push_str("    }\n");
        code.push_str("}\n");
        code
    }

    fn generate_forward_dag(&self, nodes: &[super::types::GraphNode]) -> String {
        let mut code = String::new();
        let mut var_map = HashMap::new();
        let mut placeholders = Vec::new();

        // 1. Collect placeholders and identify node return types
        let mut node_types = HashMap::new();
        for node in nodes {
            if node.op == "placeholder" {
                placeholders.push(node.name.clone());
                node_types.insert(node.name.clone(), super::types::ReturnType::Tensor);
            } else if node.op == "call_module" {
                node_types.insert(node.name.clone(), super::types::ReturnType::Tensor);
            }
            // Other ops will have their types inferred in map_fx_op
        }

        // 2. Map placeholders to arguments
        // Currently we assume the first placeholder is 'xs'
        for (i, p) in placeholders.iter().enumerate() {
            if i == 0 {
                var_map.insert(p.clone(), "xs".to_string());
            } else {
                // FEATURE: Multiple inputs
                var_map.insert(p.clone(), format!("xs{}", i));
            }
        }

        // 3. Process nodes
        for node in nodes {
            if node.op == "placeholder" {
                continue;
            }

            let sanitized = self.sanitize_name(&node.name);

            if node.op == "call_module" {
                let args: Vec<String> = node
                    .args
                    .iter()
                    .map(|a| self.resolve_fx_arg(a, &var_map))
                    .collect();

                let call_args = if args.is_empty() {
                    "/* error: no args for module call */".to_string()
                } else {
                    format!("&{}", args[0])
                };

                // Groups resolution
                let groups = self.get_groups();
                let target_sanitized = self.resolve_target(&node.target, &groups);

                code.push_str(&format!(
                    "        let {} = self.{}.forward({})?;\n",
                    sanitized, target_sanitized, call_args
                ));
                var_map.insert(node.name.clone(), sanitized);
            } else if node.op == "call_function" || node.op == "call_method" {
                let args: Vec<String> = node
                    .args
                    .iter()
                    .map(|a| self.resolve_fx_arg(a, &var_map))
                    .collect();

                let (expr, ret) = if node.op == "call_function" {
                    self.map_fx_op(
                        &node.target,
                        &args,
                        &var_map,
                        &node_types,
                        &node.args,
                        &node.kwargs,
                    )
                } else {
                    let self_var = &args[0];
                    let method_args = &args[1..];
                    self.map_fx_method(
                        &node.target,
                        self_var,
                        node.args[0].as_str().unwrap_or(""),
                        method_args,
                        &node_types,
                        &node.kwargs,
                        &var_map,
                    )
                };

                node_types.insert(node.name.clone(), ret);

                // Auto-Contiguous Heuristic
                // If this op was permute/transpose, mark it
                if node.target == "permute" || node.target == "transpose" || node.target == "t" {
                    self.permuted_vars.borrow_mut().insert(sanitized.clone());
                }

                // If this op is view/reshape, and input was permuted, inject .contiguous()
                if (node.target == "view" || node.target == "reshape") && !args.is_empty() {
                    if let Some(arg0) = args.get(0) {
                        // Check if arg0 (the variable name) is in our permuted set
                        // We need to resolve it back to the sanitized name
                        if self.permuted_vars.borrow().contains(arg0) {
                            // We can't change the generated code of the previous line easily here
                            // But we can check if the generated expr for THIS line starts with arg0
                            // Actually, map_fx_method generates "{}.view(...)", so we can just inject it there?
                            // No, map_fx_method is called above.
                            // Better: In map_fx_method, check if self_var is in permuted_vars.
                        }
                    }
                }

                code.push_str(&format!("        let {} = {};\n", sanitized, expr));
                var_map.insert(node.name.clone(), sanitized);
            } else if node.op == "output" {
                let arg0 = &node.args[0];
                if let Some(s) = arg0.as_str() {
                    if s.contains("{") && s.contains("}") {
                        let values = self.extract_values_from_dict_string(s);
                        let mapped: Vec<String> = values
                            .iter()
                            .map(|v| var_map.get(v).cloned().unwrap_or(v.clone()))
                            .collect();
                        code.push_str(&format!("        Ok(({}))\n", mapped.join(", ")));
                    } else {
                        let val = var_map.get(s).cloned().unwrap_or(s.to_string());
                        code.push_str(&format!("        Ok({})\n", val));
                    }
                } else if let Some(arr) = arg0.as_array() {
                    let mut mapped = Vec::new();
                    for v in arr {
                        if let Some(s) = v.as_str() {
                            mapped.push(var_map.get(s).cloned().unwrap_or(s.to_string()));
                        }
                    }
                    if mapped.len() > 1 {
                        code.push_str(&format!("        Ok(({}))\n", mapped.join(", ")));
                    } else {
                        code.push_str(&format!("        Ok({})\n", mapped[0]));
                    }
                } else {
                    code.push_str(&format!("        Ok({})\n", arg0));
                }
            }
        }

        code
    }

    fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
        // Check GPT2 helper first
        if let Some(init) = gpt2::generate_init(layer_name, meta, &self.config.dims) {
            return init;
        }

        // Core types
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
                    return format!(
                        "pycandle_core::layers::load_weight_norm_linear(vb.pp(\"{}\"), {}, {}, {})?",
                        layer_name, in_f, out_f, bias
                    );
                }

                let needs_transpose = meta
                    .config
                    .get("weight_shape")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        let dims: Vec<u64> = arr.iter().filter_map(|x| x.as_u64()).collect();
                        dims.len() == 2 && dims[0] == in_f_val as u64 && dims[1] == out_f_val as u64
                    })
                    .unwrap_or(false);

                if needs_transpose {
                    format!(
                        "{{ let w = weights::get_cast(&vb.pp(\"{}\"), ({}, {}), \"weight\")?.t()?; \
                         let b = {}; candle_nn::Linear::new(w, b) }}",
                        layer_name,
                        in_f,
                        out_f,
                        if bias {
                            format!(
                                "Some(weights::get_cast(&vb.pp(\"{}\"), {}, \"bias\")?)",
                                layer_name, out_f
                            )
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
                let shape_str = if shape.len() == 1 {
                    format!("{}", shape[0])
                } else {
                    format!("&{:?}", shape)
                };
                format!(
                    "candle_nn::layer_norm({}, candle_nn::LayerNormConfig {{ eps: {:e}, ..Default::default() }}, vb.pp(\"{}\"))?",
                    shape_str, eps, layer_name
                )
            }
            "LlamaRMSNorm" => {
                let eps = meta.config["variance_epsilon"].as_f64().unwrap_or(1e-5);
                let size = meta
                    .config
                    .get("weight_shape")
                    .and_then(|v| v.as_array())
                    .and_then(|a| a.last())
                    .and_then(|v| v.as_u64())
                    .map(|v| v as usize)
                    .unwrap_or(1024);
                format!(
                    "LlamaRMSNorm::load(vb.pp(\"{}\"), {}, {:e})?",
                    layer_name, size, eps
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
            "ReLU" => "ReLU".to_string(),
            "GELU" => "GELU".to_string(),
            "Sigmoid" => "Sigmoid".to_string(),
            "Tanh" => "Tanh".to_string(),
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
            "Upsample" => {
                let scale = meta.config["scale_factor"].as_u64().unwrap_or(1) as usize;
                format!("Upsample::new({})", scale)
            }
            "ReflectionPad1d" => {
                let p = meta.config["padding"].as_u64().unwrap_or(0) as usize;
                format!("ReflectionPad1d::new({})", p)
            }
            "SineGen" => {
                let harmonic_num = meta.config["harmonic_num"].as_u64().unwrap_or(0) as usize;
                let sine_amp = meta.config["sine_amp"].as_f64().unwrap_or(0.1);
                let noise_std = meta.config["noise_std"].as_f64().unwrap_or(0.003);
                let sr = meta.config["sampling_rate"].as_f64().unwrap_or(24000.0);
                let thresh = meta.config["voiced_threshold"].as_f64().unwrap_or(0.0);
                format!(
                    "SineGen::new({}, {:.6}, {:.6}, {:.1}, {:.1})",
                    harmonic_num, sine_amp, noise_std, sr, thresh
                )
            }
            _ => format!(
                "todo!(\"Implement initialization for {}\")",
                meta.module_type
            ),
        }
    }

    fn infer_linear_dims(&self, meta: &LayerMeta) -> (usize, usize) {
        let in_f = meta.config.get("in_features").and_then(|v| v.as_u64());
        let out_f = meta.config.get("out_features").and_then(|v| v.as_u64());

        if let (Some(i), Some(o)) = (in_f, out_f) {
            return (i as usize, o as usize);
        }

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
                        if let Some(obj) = arg0.as_object() {
                            if obj.len() > 1 {
                                let tensors = vec!["Tensor"; obj.len()];
                                return format!("Result<({})>", tensors.join(", "));
                            }
                        }
                        if let Some(s) = arg0.as_str() {
                            if s.contains("{") && s.contains("}") {
                                let values = self.extract_values_from_dict_string(s);
                                if values.len() > 1 {
                                    let tensors = vec!["Tensor"; values.len()];
                                    return format!("Result<({})>", tensors.join(", "));
                                }
                            }
                        }
                    }
                }
            }
        }
        "Result<Tensor>".to_string()
    }
}
