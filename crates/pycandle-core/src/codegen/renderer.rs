use super::gpt2;
use crate::LayerMeta;
use crate::codegen::ModuleNode;
use regex::Regex;
use std::collections::hash_map::DefaultHasher;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};

struct GroupInfo {
    _name: String,
    count: usize,
    module_type: String,
}

impl super::Codegen {
    fn resolve_path(&self, root: &ModuleNode, target: &str) -> String {
        let parts: Vec<&str> = target.split('.').collect();
        let mut current = root;
        let mut path = Vec::new();

        for (_, part) in parts.iter().enumerate() {
            // Check if we are at an Array index
            if let Ok(idx) = part.parse::<usize>() {
                path.push(format!("[{}]", idx));
                // array traversal
                if let ModuleNode::Array(arr) = current {
                    if idx < arr.len() {
                        current = &arr[idx];
                    } else {
                        // Fallback or error?
                        // If index out of bounds structurally (unlikely in valid trace),
                        // we assume homogenous array and just take the first one for type logic,
                        // but for PATH resolution, we just accept the index.
                        // But we need 'current' to advance.
                        if !arr.is_empty() {
                            current = &arr[0]; // best effort for traversing type info if valid
                        }
                    }
                }
                continue;
            }

            // Normal field traversal
            path.push(format!(".{}", self.sanitize_name(part)));

            match current {
                ModuleNode::Struct(children) => {
                    if let Some(child) = children.get(*part) {
                        current = child;
                    }
                }
                _ => {}
            }
        }

        // Join path, remove leading dot if exists (it will be tacked onto 'self')
        let full = path.join("");
        if full.starts_with('.') {
            full[1..].to_string()
        } else {
            full
        }
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

        let module_tree = self.build_module_tree();

        // Generate Model struct and sub-structs recursively
        // This populates the registry
        let root_type_name = self.render_node(model_name, &module_tree);

        // Dump all definitions in order
        for def in self.struct_definitions.borrow().iter() {
            code.push_str(def);
        }

        // Generate Root Forward Impl (always unique to root)
        code.push_str(&self.render_root_forward(&root_type_name, &module_tree));

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

    // Recursive Node Renderer with Deduplication
    fn render_node(&self, type_name: &str, node: &ModuleNode) -> String {
        match node {
            ModuleNode::Struct(children) => {
                // 1. Recurse children first to get their confirmed type names
                let mut child_types = HashMap::new();
                for (name, child) in children {
                    let child_type_name = match child {
                        ModuleNode::Leaf(_) => String::new(), // Leaves don't have generated types
                        ModuleNode::Struct(_) => {
                            let proposed = format!("{}{}", type_name, self.camel_case(name));
                            self.render_node(&proposed, child)
                        }
                        ModuleNode::Array(arr) => {
                            if !arr.is_empty() {
                                match &arr[0] {
                                    ModuleNode::Struct(_) => {
                                        let proposed = if name.ends_with('s') {
                                            // hacky plural de-pluralization? No, just append
                                            format!("{}{}", type_name, self.camel_case(name))
                                        } else {
                                            format!("{}{}", type_name, self.camel_case(name))
                                        };
                                        self.render_node(&proposed, &arr[0])
                                    }
                                    _ => String::new(),
                                }
                            } else {
                                String::new()
                            }
                        }
                    };
                    if !child_type_name.is_empty() {
                        child_types.insert(name.clone(), child_type_name);
                    }
                }

                // 2. Generate Struct Definition
                let mut struct_code = String::new();
                struct_code.push_str(&format!(
                    "#[derive(Clone, Debug)]\npub struct {} {{\n",
                    type_name
                )); // Added Debug, Clone
                for (name, child) in children {
                    let field_name = self.sanitize_name(name);
                    let type_str = match child {
                        ModuleNode::Leaf(meta) => self.map_type(&meta.module_type).into(),
                        ModuleNode::Struct(_) => child_types[name].clone(),
                        ModuleNode::Array(arr) => {
                            if arr.is_empty() {
                                "Vec<Tensor> /* Empty Array */".to_string()
                            } else {
                                let inner = match &arr[0] {
                                    ModuleNode::Leaf(meta) => {
                                        self.map_type(&meta.module_type).into()
                                    }
                                    ModuleNode::Struct(_) => child_types[name].clone(),
                                    ModuleNode::Array(_) => {
                                        "Vec<Tensor> /* Nested Array */".to_string()
                                    }
                                };
                                format!("Vec<{}>", inner)
                            }
                        }
                    };

                    if let ModuleNode::Leaf(meta) = child {
                        if !meta.input_shapes.is_empty() {
                            struct_code
                                .push_str(&format!("    /// Input: {:?}\n", meta.input_shapes));
                        }
                    }
                    struct_code.push_str(&format!("    pub {}: {},\n", field_name, type_str));
                }

                // Add Config to root? No, we'll handle config in load manually for now or just generic.
                // If it's the root, we might want it.
                // But deduplication breaks "Is Root" concept.
                // Only the actual usage knows.
                // We'll standardise: `load` takes `&Config` (except root wrapper which we'll handle outside).

                struct_code.push_str("}\n\n");

                // 3. Generate Impl Load
                let mut impl_code = String::new();
                impl_code.push_str(&format!("impl {} {{\n", type_name));
                impl_code.push_str(
                    "    pub fn load(vb: VarBuilder, config: &Config) -> Result<Self> {\n",
                );
                impl_code.push_str("        Ok(Self {\n");

                for (name, child) in children {
                    let field_name = self.sanitize_name(name);
                    match child {
                        ModuleNode::Leaf(meta) => {
                            let init = self.generate_init(name, meta);
                            impl_code.push_str(&format!("            {}: {},\n", field_name, init));
                        }
                        ModuleNode::Struct(_) => {
                            let child_type = &child_types[name];
                            impl_code.push_str(&format!(
                                "            {}: {}::load(vb.pp(\"{}\"), config)?,\n",
                                field_name, child_type, name
                            ));
                        }
                        ModuleNode::Array(arr) => {
                            if arr.is_empty() {
                                impl_code.push_str(&format!(
                                    "            {}: Vec::new(),\n",
                                    field_name
                                ));
                            } else {
                                impl_code.push_str(&format!(
                                    "            {}: (0..{}).map(|i| {{\n",
                                    field_name,
                                    arr.len()
                                ));
                                match &arr[0] {
                                    ModuleNode::Leaf(meta) => {
                                        let replacement = format!(
                                            "vb.pp(&format!(\"{{}}.{{}}\", \"{}\", i))",
                                            name
                                        );
                                        let template = self.generate_init("PLACEHOLDER", meta);
                                        let fixed = template
                                            .replace("vb.pp(\"PLACEHOLDER\")", &replacement);
                                        impl_code.push_str(&format!("                {}\n", fixed));
                                    }
                                    ModuleNode::Struct(_) => {
                                        let child_type = &child_types[name];
                                        impl_code.push_str(&format!("                {}::load(vb.pp(&format!(\"{{}}.{{}}\", \"{}\", i)), config)?\n", child_type, name));
                                    }
                                    _ => impl_code.push_str("todo!(),\n"),
                                }
                                impl_code
                                    .push_str("            }).collect::<Result<Vec<_>>>()?,\n");
                            }
                        }
                    }
                }
                impl_code.push_str("        })\n");
                impl_code.push_str("    }\n");
                impl_code.push_str("}\n\n");

                // 4. Compute Hash
                // Note: We need to anonymize the struct name in the code before hashing?
                // OR: We include the body, but wait, the body contains `child_types`.
                // If child A and child B are identical, `child_types` will preserve that.
                // The Type Name *inside* the struct definition must be generic for the hash?
                // No, because the child type name IS the identifier.
                // `Box<BlockA>` vs `Box<BlockB>`.
                // If `BlockA` was deduplicated to `Block`, then `BlockB` also became `Block`.
                // So the *content* is identical.
                // BUT the `struct Name {` ... `impl Name` parts need to be normalized before hashing.
                // Replace `type_name` with "SELF" for hashing.

                let anon_struct_code = struct_code.replace(type_name, "SELF_STRUCT_NAME");
                let anon_impl_code = impl_code.replace(type_name, "SELF_STRUCT_NAME");

                let mut hasher = DefaultHasher::new();
                anon_struct_code.hash(&mut hasher);
                anon_impl_code.hash(&mut hasher);
                let signature = hasher.finish();

                // 5. Check Registry
                let mut registry = self.struct_registry.borrow_mut();
                if let Some(existing_name) = registry.get(&signature) {
                    return existing_name.clone();
                }

                // 6. Register New
                registry.insert(signature, type_name.to_string());

                // Add to definitions
                let full_code = format!("{}{}", struct_code, impl_code);
                self.struct_definitions.borrow_mut().push(full_code);

                type_name.to_string()
            }
            _ => String::new(),
        }
    }

    // CamelCase Helper
    fn camel_case(&self, s: &str) -> String {
        let mut result = String::new();
        let mut capitalize = true;
        for c in s.chars() {
            if c == '_' || c == '.' {
                capitalize = true;
            } else {
                if capitalize {
                    result.push(c.to_ascii_uppercase());
                    capitalize = false;
                } else {
                    result.push(c);
                }
            }
        }
        result
    }

    fn render_root_forward(&self, type_name: &str, root_node: &ModuleNode) -> String {
        let mut code = String::new();
        code.push_str(&format!("impl {} {{\n", type_name));

        // 1. load_from_hub (Root Specific)
        code.push_str("    pub fn load_from_hub(repo: &str, revision: &str, device: &Device, config: Config) -> Result<Self> {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo.to_string()).with_revision(revision.to_string());
        let path = repo.get(\"model.safetensors\")?;
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[path], device.clone())? };
        Self::load(vb, &config)
    }
");

        // 2. forward (DAG or Sequential)
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
            let nodes = &self.graph_nodes;
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

                    // Recursive Path Resolution
                    let target_sanitized = self.resolve_path(root_node, &node.target);

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
                    if node.target == "permute" || node.target == "transpose" || node.target == "t"
                    {
                        self.permuted_vars.borrow_mut().insert(sanitized.clone());
                    }

                    if (node.target == "view" || node.target == "reshape") && !args.is_empty() {
                        if let Some(arg0) = args.get(0) {
                            if self.permuted_vars.borrow().contains(arg0) {
                                // Logic implicit via map_fx_method?
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
        } else {
            code.push_str("        // TODO: Sequential fallback with recursive tree\n");
            code.push_str(
                "        todo!(\"Sequential fallback not implemented for recursive tree\")\n",
            );
        }

        code.push_str("    }\n");
        code.push_str("}\n");

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
