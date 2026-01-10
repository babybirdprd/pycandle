use super::types::ReturnType;
use std::collections::HashMap;

impl super::Codegen {
    pub fn map_fx_op(
        &self,
        target: &str,
        args: &[String],
        var_map: &HashMap<String, String>,
        node_types: &HashMap<String, ReturnType>,
        _raw_args: &[serde_json::Value], // Needed to check for literals
        kwargs: &HashMap<String, serde_json::Value>,
    ) -> (String, ReturnType) {
        let target_lower = target.to_lowercase();

        // Binary ops are now handled in the main match arm or further down
        // to avoid duplicate blocks and allow for specific op handling like addmm.

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
                    // Vec indexing: x[0] or x[-1] or x[slice(...)]
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
                    } else if idx.contains("slice(") {
                        // Handled by the generic slice parsing logic that we want to unify
                        let slice_str = idx.clone();
                        let (expr, ret) = self.parse_vec_slice(&args[0], &slice_str);
                        return (expr, ret);
                    } else {
                        return (
                            format!("{}[{}].clone()", args[0], args[1]),
                            ReturnType::Primitive,
                        );
                    }
                }
                ReturnType::Tensor => {
                    if idx.contains("slice(")
                        || idx == "None"
                        || idx.starts_with("&[")
                        || idx.starts_with("vec![")
                    {
                        // Map to .i() for indexing/slicing
                        let mut cleaned = idx.trim().to_string();
                        loop {
                            let start_len = cleaned.len();
                            if cleaned.starts_with("(") && cleaned.ends_with(")") {
                                cleaned = cleaned[1..cleaned.len() - 1].trim().to_string();
                            }
                            if cleaned.starts_with("&[") {
                                cleaned = cleaned[2..cleaned.len() - 1].trim().to_string();
                            }
                            if cleaned.starts_with("vec![") {
                                cleaned = cleaned[5..cleaned.len() - 1].trim().to_string();
                            }
                            if cleaned.len() == start_len {
                                break;
                            }
                        }

                        let items: Vec<String> = self
                            .split_indices(&cleaned)
                            .into_iter()
                            .enumerate()
                            .map(|(i, s)| self.parse_slice_item(s.trim(), &args[0], i))
                            .collect();

                        return (
                            format!(
                                "pycandle_core::ops::index(&{}, vec![{}])?",
                                args[0],
                                items.join(", ")
                            ),
                            ReturnType::Tensor,
                        );
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
                ReturnType::FInfo => {
                    return (
                        format!("todo!(/* FInfo indexing on {} */)", args[0]),
                        ReturnType::FInfo,
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
                let dim = if let Some(v) = kwargs.get("dim") {
                    self.resolve_fx_arg(v, var_map)
                } else if args.len() > 2 {
                    args[2].clone()
                } else {
                    "0".to_string()
                };
                (
                    format!("{}.chunk({}, {})?", args[0], chunks, dim),
                    ReturnType::Vec,
                )
            }
            "torch.split" | "split" => {
                let split_size = args.get(1).map(|s| s.as_str()).unwrap_or("1");
                let dim = if let Some(v) = kwargs.get("dim") {
                    self.resolve_fx_arg(v, var_map)
                } else if args.len() > 2 {
                    args[2].clone()
                } else {
                    "0".to_string()
                };
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
            "torch.addmm" | "addmm" | "aten.addmm" => {
                // addmm(bias, a, b) -> (a @ b) + bias
                // a is typically input, b is typically weight (out, in) in Candle, so we need b.t()
                if args.len() >= 3 {
                    let w = &args[2];
                    // Heuristic: if b is a weight attribute, it likely needs transposition for matmul
                    let w_suffix = if w.contains("weight") || w.contains("w_") {
                        ".t()?"
                    } else {
                        ""
                    };
                    return (
                        format!(
                            "{}.matmul(&{}{})?.broadcast_add(&{})?",
                            args[1], w, w_suffix, args[0]
                        ),
                        ReturnType::Tensor,
                    );
                }
                (
                    format!("todo!(/* addmm with {} args */)", args.len()),
                    ReturnType::Tensor,
                )
            }
            // FEATURE: Comparisons
            "torch.lt" | "lt" | "_operator.lt" => {
                // Check types for primitive comparison
                let type0 = node_types
                    .get(&args[0])
                    .copied()
                    .unwrap_or(ReturnType::Tensor);
                let type1 = node_types
                    .get(&args[1])
                    .copied()
                    .unwrap_or(ReturnType::Tensor);

                if type0 == ReturnType::Primitive || type1 == ReturnType::Primitive {
                    (
                        format!("({} < {})", args[0], args[1]),
                        ReturnType::Primitive,
                    )
                } else {
                    // BROADCAST FIX: Ensure shapes are compatible for Candle comparison
                    (
                        format!("pycandle_core::ops::lt(&{}, &{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    )
                }
            }
            "torch.gt" | "gt" | "_operator.gt" => {
                let type0 = node_types
                    .get(&args[0])
                    .copied()
                    .unwrap_or(ReturnType::Tensor);
                let type1 = node_types
                    .get(&args[1])
                    .copied()
                    .unwrap_or(ReturnType::Tensor);

                if type0 == ReturnType::Primitive || type1 == ReturnType::Primitive {
                    (
                        format!("({} > {})", args[0], args[1]),
                        ReturnType::Primitive,
                    )
                } else {
                    // BROADCAST FIX: Ensure shapes are compatible for Candle comparison
                    (
                        format!("pycandle_core::ops::gt(&{}, &{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    )
                }
            }
            "slice" => {
                let start = args.get(0).map(|s| s.as_str()).unwrap_or("None");
                let stop = args.get(1).map(|s| s.as_str()).unwrap_or("None");

                let start_s = if start == "None" || start.is_empty() {
                    "None".to_string()
                } else {
                    format!("Some({})", start)
                };
                let stop_s = if stop == "None" || stop.is_empty() {
                    "None".to_string()
                } else {
                    format!("Some({})", stop)
                };

                (
                    format!(
                        "pycandle_core::ops::IndexItem::Slice({}, {})",
                        start_s, stop_s
                    ),
                    ReturnType::Primitive,
                )
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
                        "pycandle_core::ops::scaled_dot_product_attention(&{}, &{}, &{}, {}, {:.1}, {}, {})?",
                        q, k, v, attn_mask, dropout_p, is_causal, scale
                    ),
                    ReturnType::Tensor,
                )
            }
            // FEATURE: builtins.getattr
            "builtins.getattr" => {
                let obj = &args[0];
                let attr = args[1].trim_matches('"');

                // Check if obj is an FInfo object
                let obj_type = var_map
                    .get(obj)
                    .and_then(|v| node_types.get(v))
                    .cloned()
                    .unwrap_or(ReturnType::Tensor);
                if matches!(obj_type, ReturnType::FInfo) {
                    match attr {
                        "min" => return ("f32::MIN".to_string(), ReturnType::Primitive),
                        "max" => return ("f32::MAX".to_string(), ReturnType::Primitive),
                        _ => {
                            return (
                                format!("todo!(/* FInfo attr {} */)", attr),
                                ReturnType::Primitive,
                            );
                        }
                    }
                }

                match attr {
                    "device" => (format!("{}.device()", obj), ReturnType::Primitive),
                    "dtype" => (format!("{}.dtype()", obj), ReturnType::Primitive),
                    "shape" => (format!("{}.dims().to_vec()", obj), ReturnType::Vec),
                    _ => (
                        format!("todo!(/* getattr {} on {} */)", attr, obj),
                        ReturnType::Primitive,
                    ),
                }
            }
            // FEATURE: torch.finfo
            "torch.finfo" => {
                // We return a dummy because the value is only used for getattr
                ("()".to_string(), ReturnType::FInfo)
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
                        format!("Tensor::arange(0i64, {} as i64, {})?", args[0], dev),
                        ReturnType::Tensor,
                    )
                } else if args.len() >= 2 {
                    (
                        format!(
                            "Tensor::arange({} as i64, {} as i64, {})?",
                            args[0], args[1], dev
                        ),
                        ReturnType::Tensor,
                    )
                } else {
                    (
                        "Tensor::arange(0i64, 1i64, Device::Cpu)?".to_string(),
                        ReturnType::Tensor,
                    )
                }
            }
            "torch.full" => {
                let shape = args[0].clone();
                let fill_value = if let Some(v) = kwargs.get("fill_value") {
                    self.resolve_fx_arg(v, var_map)
                } else if args.len() > 1 {
                    args[1].clone()
                } else {
                    "0.0".to_string()
                };
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
            "torch.masked_fill" | "masked_fill" => {
                let tensor = &args[0];
                let mask = &args[1];
                let mut value = args[2].clone();
                if !value.contains('.') && !value.contains('e') && value.parse::<f64>().is_ok() {
                    value.push_str(".0");
                }
                // Force .0 if it looks optionally like an integer
                if let Ok(_) = value.parse::<i64>() {
                    if !value.contains('.') {
                        value.push_str(".0");
                    }
                }
                (
                    format!(
                        "pycandle_core::ops::masked_fill(&{}, &{}, {})?",
                        tensor, mask, value
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
                        // Vec indexing: x[0] -> returns a single element (Primitive)
                        if let Ok(i) = idx.parse::<usize>() {
                            (format!("{}[{}].clone()", args[0], i), ReturnType::Primitive)
                        } else {
                            (
                                format!("{}.get({})?", args[0], args[1]),
                                ReturnType::Primitive,
                            )
                        }
                    }
                    ReturnType::Tensor => {
                        if idx.contains("slice(") || idx == "None" || idx.starts_with("&[") {
                            // Map to .i() for indexing/slicing
                            let mut cleaned = idx.trim().to_string();
                            loop {
                                let start_len = cleaned.len();
                                if cleaned.starts_with("(") && cleaned.ends_with(")") {
                                    cleaned = cleaned[1..cleaned.len() - 1].trim().to_string();
                                }
                                if cleaned.starts_with("&[") {
                                    cleaned = cleaned[2..cleaned.len() - 1].trim().to_string();
                                }
                                if cleaned.starts_with("vec![") {
                                    cleaned = cleaned[5..cleaned.len() - 1].trim().to_string();
                                }
                                if cleaned.len() == start_len {
                                    break;
                                }
                            }

                            let items: Vec<String> = self
                                .split_indices(&cleaned)
                                .into_iter()
                                .enumerate()
                                .map(|(i, s)| self.parse_slice_item(s.trim(), &args[0], i))
                                .collect();

                            (
                                format!(
                                    "pycandle_core::ops::index(&{}, vec![{}])?",
                                    args[0],
                                    items.join(", ")
                                ),
                                ReturnType::Tensor,
                            )
                        } else {
                            (format!("{}.get({})?", args[0], args[1]), ReturnType::Tensor)
                        }
                    }
                    ReturnType::Primitive => (
                        format!("todo!(/* primitive indexing on {} */)", args[0]),
                        ReturnType::Primitive,
                    ),
                    ReturnType::FInfo => (
                        format!("todo!(/* FInfo indexing on {} */)", args[0]),
                        ReturnType::Primitive,
                    ),
                }
            }
            _ => {
                if target_lower.contains("add") && args.len() >= 2 {
                    let type0 = _raw_args
                        .get(0)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);
                    let type1 = _raw_args
                        .get(1)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);

                    if type0 == ReturnType::Primitive || type1 == ReturnType::Primitive {
                        let a0 = if args[0].parse::<i64>().is_ok() {
                            format!("{}usize", args[0])
                        } else {
                            args[0].clone()
                        };
                        let a1 = if args[1].parse::<i64>().is_ok() {
                            format!("{}usize", args[1])
                        } else {
                            args[1].clone()
                        };
                        return (format!("({} + {})", a0, a1), ReturnType::Primitive);
                    }

                    let is_scalar_0 = args[0].parse::<f64>().is_ok();
                    let is_scalar_1 = args[1].parse::<f64>().is_ok();

                    if is_scalar_0 || is_scalar_1 {
                        let scalar = if is_scalar_0 { &args[0] } else { &args[1] };
                        let tensor = if is_scalar_0 { &args[1] } else { &args[0] };

                        let mut s_val = scalar.clone();
                        if !s_val.contains('.') && !s_val.contains('e') {
                            s_val.push_str(".0");
                        }
                        return (
                            format!("{}.affine({}, 0.0)?", tensor, s_val),
                            ReturnType::Tensor,
                        );
                    }

                    if self.is_vec_op(&args[0], &args[1], node_types, var_map) {
                        let mut arg0 = args[0].clone();
                        if arg0.contains("-1") && arg0.starts_with("vec![") {
                            arg0 = arg0.replace("-1", "-1isize");
                        }
                        return (
                            format!(
                                "{{ let mut v: Vec<isize> = {}.iter().map(|&x| x as isize).collect(); v.extend({}.iter().map(|&x| x as isize)); v }}",
                                arg0, args[1]
                            ),
                            ReturnType::Vec,
                        );
                    }

                    return (
                        format!("{}.broadcast_add(&{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }
                if target_lower.contains("sub") && args.len() >= 2 {
                    let type0 = _raw_args
                        .get(0)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);
                    let type1 = _raw_args
                        .get(1)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);

                    if type0 == ReturnType::Primitive || type1 == ReturnType::Primitive {
                        return (
                            format!("({} - {})", args[0], args[1]),
                            ReturnType::Primitive,
                        );
                    }

                    let is_scalar_0 = args[0].parse::<f64>().is_ok();
                    let is_scalar_1 = args[1].parse::<f64>().is_ok();
                    if is_scalar_0 || is_scalar_1 {
                        let scalar = if is_scalar_0 { &args[0] } else { &args[1] };
                        let tensor = if is_scalar_0 { &args[1] } else { &args[0] };

                        let mut s_val = scalar.clone();
                        if !s_val.contains('.') && !s_val.contains('e') {
                            s_val.push_str(".0");
                        }

                        let expr = if is_scalar_0 {
                            format!("Ok({} - {})?", scalar, tensor)
                        } else {
                            format!("{}.affine(1.0f64, -{}f64)?", tensor, s_val)
                        };
                        return (expr, ReturnType::Tensor);
                    }
                    return (
                        format!("{}.broadcast_sub(&{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }
                if target_lower.contains("mul") && args.len() >= 2 {
                    let type0 = _raw_args
                        .get(0)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);
                    let type1 = _raw_args
                        .get(1)
                        .and_then(|v| v.as_str())
                        .and_then(|s| node_types.get(s))
                        .copied()
                        .unwrap_or(ReturnType::Tensor);

                    if type0 == ReturnType::Primitive || type1 == ReturnType::Primitive {
                        return (
                            format!("({} * {})", args[0], args[1]),
                            ReturnType::Primitive,
                        );
                    }

                    let is_scalar_0 = args[0].parse::<f64>().is_ok();
                    let is_scalar_1 = args[1].parse::<f64>().is_ok();
                    if is_scalar_0 || is_scalar_1 {
                        let scalar = if is_scalar_0 { &args[0] } else { &args[1] };
                        let tensor = if is_scalar_0 { &args[1] } else { &args[0] };

                        let mut s_val = scalar.clone();
                        if !s_val.contains('.') && !s_val.contains('e') {
                            s_val.push_str(".0");
                        }
                        return (
                            format!("{}.affine({}f64, 0.0f64)?", tensor, s_val),
                            ReturnType::Tensor,
                        );
                    }
                    return (
                        format!("{}.broadcast_mul(&{})?", args[0], args[1]),
                        ReturnType::Tensor,
                    );
                }
                (
                    format!("todo!(/* op: {} on {:?} */)", target, args),
                    ReturnType::Tensor,
                )
            }
        }
    }

    pub fn resolve_fx_arg(
        &self,
        arg: &serde_json::Value,
        var_map: &HashMap<String, String>,
    ) -> String {
        match arg {
            serde_json::Value::String(s) => {
                // If it's a variable in the local scope, use it
                if let Some(v) = var_map.get(s) {
                    return v.clone();
                }

                // If it's a parameter or buffer name (from manifest), prefix it with self.
                if self.manifest.contains_key(s) {
                    return format!("self.{}", self.sanitize_name(s));
                }

                // Check sanitized matches
                for key in self.manifest.keys() {
                    if self.sanitize_name(key) == self.sanitize_name(s) {
                        return format!("self.{}", self.sanitize_name(key));
                    }
                }

                s.clone()
            }
            serde_json::Value::Number(n) => n.to_string(),
            serde_json::Value::Bool(b) => b.to_string(),
            serde_json::Value::Array(a) => {
                let items: Vec<String> =
                    a.iter().map(|v| self.resolve_fx_arg(v, var_map)).collect();
                if items.is_empty() {
                    "&[]".to_string()
                } else if items.iter().all(|s| s.parse::<f64>().is_ok()) {
                    format!("&[{}]", items.join(", "))
                } else {
                    format!("vec![{}]", items.join(", "))
                }
            }
            serde_json::Value::Null => "None".to_string(),
            _ => arg.to_string(),
        }
    }

    pub fn map_fx_method(
        &self,
        method: &str,
        self_var: &str,
        self_var_name: &str,
        args: &[String],
        node_types: &HashMap<String, ReturnType>,
        kwargs: &HashMap<String, serde_json::Value>,
        _var_map: &HashMap<String, String>,
    ) -> (String, ReturnType) {
        let _src_type = node_types
            .get(self_var_name)
            .cloned()
            .unwrap_or(ReturnType::Tensor);

        match method {
            "view" | "reshape" => {
                let mapped_args: Vec<String> = args
                    .iter()
                    .map(|s| {
                        if s == "-1" {
                            "-1isize".to_string()
                        } else if s.parse::<i64>().is_ok() {
                            format!("{}isize", s)
                        } else {
                            format!("{} as isize", s)
                        }
                    })
                    .collect();

                let is_single_vec = if args.len() == 1 {
                    let arg_name = args[0].trim();
                    let fx_name = node_types.keys().find(|k| {
                        let sanitized = k.replace(".", "_");
                        sanitized == arg_name || k.as_str() == arg_name
                    });
                    fx_name
                        .and_then(|k| node_types.get(k))
                        .map(|t| *t == ReturnType::Vec)
                        .unwrap_or(false)
                } else {
                    false
                };

                if is_single_vec {
                    (
                        format!(
                            "pycandle_core::ops::reshape(&{}, &{}.iter().map(|&x| x as isize).collect::<Vec<_>>())?",
                            self_var, args[0]
                        ),
                        ReturnType::Tensor,
                    )
                } else if args.len() == 1 && args[0].starts_with("vec![") {
                    (
                        format!("pycandle_core::ops::reshape(&{}, &{})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                } else if args.len() == 1
                    && node_types
                        .get(&args[0])
                        .map(|t| *t == ReturnType::Vec)
                        .unwrap_or(false)
                {
                    (
                        format!(
                            "pycandle_core::ops::reshape(&{}, &{}.iter().map(|&x| x as isize).collect::<Vec<_>>())?",
                            self_var, args[0]
                        ),
                        ReturnType::Tensor,
                    )
                } else if args.len() == 1 && args[0].starts_with("&[") {
                    (
                        format!("pycandle_core::ops::reshape(&{}, &{})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                } else {
                    (
                        format!(
                            "pycandle_core::ops::reshape(&{}, &vec![{}])?",
                            self_var,
                            mapped_args.join(", ")
                        ),
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
            "to" => {
                if args.is_empty() {
                    let dtype_val = kwargs.get("dtype").and_then(|v| v.as_str());
                    let device_val = kwargs.get("device").and_then(|v| v.as_str());

                    if let Some(dt) = dtype_val {
                        (
                            format!("{}.to_dtype({})?", self_var, dt),
                            ReturnType::Tensor,
                        )
                    } else if let Some(dv) = device_val {
                        (
                            format!("{}.to_device(&{})?", self_var, dv),
                            ReturnType::Tensor,
                        )
                    } else {
                        (self_var.to_string(), ReturnType::Tensor)
                    }
                } else if args.len() == 1 {
                    if args[0].contains("DType") {
                        (
                            format!("{}.to_dtype({})?", self_var, args[0]),
                            ReturnType::Tensor,
                        )
                    } else if args[0].contains("Device") {
                        (
                            format!("{}.to_device(&{})?", self_var, args[0]),
                            ReturnType::Tensor,
                        )
                    } else {
                        (
                            format!("{}.to_dtype({})?", self_var, args[0]),
                            ReturnType::Tensor,
                        )
                    }
                } else {
                    (
                        format!("{}.to_dtype({})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                }
            }
            "expand" => {
                let shape_arg = if args.len() == 1 {
                    args[0].clone()
                } else {
                    format!("({})", args.join(", "))
                };
                (
                    format!("{}.broadcast_as({})?", self_var, shape_arg),
                    ReturnType::Tensor,
                )
            }
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
                let dim = args.get(1).map(|s| s.as_str()).unwrap_or("0");
                (
                    format!(
                        "pycandle_core::ops::split(&{}, {} as usize, {} as usize)?",
                        self_var, args[0], dim
                    ),
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
                        format!(
                            "pycandle_core::ops::dim(&{}, {} as isize)?",
                            self_var, args[0]
                        ),
                        ReturnType::Tensor,
                    )
                }
            }
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
            "lt" | "_operator.lt" => {
                let self_type = node_types
                    .get(self_var_name)
                    .copied()
                    .unwrap_or(ReturnType::Tensor);
                let arg_is_int = args[0].parse::<i64>().is_ok() || args[0].parse::<usize>().is_ok();

                if self_type == ReturnType::Primitive || arg_is_int {
                    (
                        format!("({} < {})", self_var, args[0]),
                        ReturnType::Primitive,
                    )
                } else {
                    (
                        format!("pycandle_core::ops::lt(&{}, &{})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                }
            }
            "gt" | "_operator.gt" => {
                let self_type = node_types
                    .get(self_var_name)
                    .copied()
                    .unwrap_or(ReturnType::Tensor);
                let arg_is_int = args[0].parse::<i64>().is_ok() || args[0].parse::<usize>().is_ok();

                if self_type == ReturnType::Primitive || arg_is_int {
                    (
                        format!("({} > {})", self_var, args[0]),
                        ReturnType::Primitive,
                    )
                } else {
                    (
                        format!("pycandle_core::ops::gt(&{}, &{})?", self_var, args[0]),
                        ReturnType::Tensor,
                    )
                }
            }
            "masked_fill" | "masked_fill_" => {
                let mask = &args[0];
                let mut value = args[1].clone();
                if !value.contains('.') && !value.contains('e') && value.parse::<f64>().is_ok() {
                    value.push_str(".0");
                }
                if let Ok(_) = value.parse::<i64>() {
                    if !value.contains('.') {
                        value.push_str(".0");
                    }
                }

                (
                    format!(
                        "pycandle_core::ops::masked_fill(&{}, &{}, {})?",
                        self_var, mask, value
                    ),
                    ReturnType::Tensor,
                )
            }
            _ => (
                format!("todo!(/* method: {} on {} */)", method, self_var),
                ReturnType::Tensor,
            ),
        }
    }

    pub fn is_vec_op(
        &self,
        a: &str,
        b: &str,
        node_types: &HashMap<String, ReturnType>,
        var_map: &HashMap<String, String>,
    ) -> bool {
        let a_name = var_map
            .iter()
            .find(|(_, v)| *v == &a)
            .map(|(k, _)| k.as_str())
            .unwrap_or(a);
        let b_name = var_map
            .iter()
            .find(|(_, v)| *v == &b)
            .map(|(k, _)| k.as_str())
            .unwrap_or(b);

        let a_type = node_types
            .get(a_name)
            .cloned()
            .unwrap_or(ReturnType::Tensor);
        let b_type = node_types
            .get(b_name)
            .cloned()
            .unwrap_or(ReturnType::Tensor);

        a_type == ReturnType::Vec || b_type == ReturnType::Vec
    }

    pub fn parse_vec_slice(&self, vec_name: &str, slice_str: &str) -> (String, ReturnType) {
        let content = slice_str
            .strip_prefix("slice(")
            .and_then(|s| s.strip_suffix(")"))
            .unwrap_or(slice_str);
        let parts: Vec<&str> = content.split(",").map(|s| s.trim()).collect();
        let start = parts.get(0).copied().unwrap_or("None");
        let stop = parts.get(1).copied().unwrap_or("None");

        let start_str = if start == "None" {
            "".to_string()
        } else {
            start.to_string()
        };
        let stop_str = if stop == "None" {
            "".to_string()
        } else {
            if let Ok(v) = stop.parse::<isize>() {
                if v < 0 {
                    format!("{}.len() - {}", vec_name, v.abs())
                } else {
                    v.to_string()
                }
            } else {
                stop.to_string()
            }
        };
        (
            format!("{}[{}..{}].to_vec()", vec_name, start_str, stop_str),
            ReturnType::Vec,
        )
    }

    pub fn extract_values_from_dict_string(&self, s: &str) -> Vec<String> {
        let mut values = Vec::new();
        let content = s.trim_matches(|c| c == '{' || c == '}');
        for part in content.split(',') {
            if let Some(pos) = part.find(':') {
                let val = part[pos + 1..].trim();
                values.push(val.to_string());
            }
        }
        values
    }

    pub fn split_indices(&self, s: &str) -> Vec<String> {
        let mut result = Vec::new();
        let mut current = String::new();
        let mut bracket_level = 0;
        let mut paren_level = 0;

        for c in s.chars() {
            match c {
                '[' => bracket_level += 1,
                ']' => bracket_level -= 1,
                '(' => paren_level += 1,
                ')' => paren_level -= 1,
                ',' if bracket_level == 0 && paren_level == 0 => {
                    result.push(current.trim().to_string());
                    current = String::new();
                    continue;
                }
                _ => {}
            }
            current.push(c);
        }
        if !current.trim().is_empty() {
            result.push(current.trim().to_string());
        }
        result
    }

    pub fn parse_slice_item(&self, item: &str, _tensor_name: &str, _dim_idx: usize) -> String {
        if item == "None" {
            return "pycandle_core::ops::IndexItem::NewAxis".to_string();
        }
        if item == "..." {
            return "pycandle_core::ops::IndexItem::Ellipsis".to_string();
        }

        if item.starts_with("slice(") {
            let content = &item[6..item.len() - 1];
            let parts: Vec<&str> = content.split(',').map(|s| s.trim()).collect();
            let start = parts.get(0).copied().unwrap_or("None");
            let stop = parts.get(1).copied().unwrap_or("None");

            let start_s = if start == "None" {
                "None".to_string()
            } else {
                format!("Some({})", start)
            };
            let stop_s = if stop == "None" {
                "None".to_string()
            } else {
                format!("Some({})", stop)
            };

            return format!(
                "pycandle_core::ops::IndexItem::Slice({}, {})",
                start_s, stop_s
            );
        }

        if let Ok(_) = item.parse::<isize>() {
            return format!("pycandle_core::ops::IndexItem::Single({})", item);
        }

        // If it's a variable or expression
        format!("pycandle_core::ops::IndexItem::Single({} as isize)", item)
    }
}
