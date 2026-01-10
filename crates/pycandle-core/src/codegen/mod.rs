use crate::LayerMeta;
use std::collections::HashMap;

pub mod analyzer;
pub mod config;
pub mod einsum;
pub mod gpt2;
pub mod ops;
pub mod renderer;
pub mod types;
pub mod utils;

pub use types::*;

pub struct Codegen {
    pub manifest: HashMap<String, LayerMeta>,
    pub hints: Option<HashMap<String, usize>>,
    pub graph_nodes: Vec<GraphNode>,
    pub stateful: bool,
    pub permuted_vars: std::cell::RefCell<std::collections::HashSet<String>>,
    pub struct_registry: std::cell::RefCell<HashMap<u64, String>>,
    pub struct_definitions: std::cell::RefCell<Vec<String>>,
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
            permuted_vars: std::cell::RefCell::new(std::collections::HashSet::new()),
            struct_registry: std::cell::RefCell::new(HashMap::new()),
            struct_definitions: std::cell::RefCell::new(Vec::new()),
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

    fn should_be_leaf(&self, module_type: &str) -> bool {
        matches!(
            module_type,
            "Linear"
                | "Conv1d"
                | "Conv2d"
                | "ConvTranspose1d"
                | "ParametrizedConv1d"
                | "ParametrizedLinear"
        )
    }

    pub fn build_module_tree(&self) -> ModuleNode {
        let mut root = std::collections::BTreeMap::new();
        for (name, meta) in &self.manifest {
            let parts: Vec<&str> = name.split('.').collect();
            self.insert_into_tree(&mut root, &parts, meta.clone());
        }
        let root = self.compress_arrays(ModuleNode::Struct(root));
        self.flatten_wrappers(root)
    }

    fn flatten_wrappers(&self, node: ModuleNode) -> ModuleNode {
        match node {
            ModuleNode::Struct(mut children) => {
                // 1. Recurse first
                for (_, child) in children.iter_mut() {
                    let new_child = self.flatten_wrappers(child.clone());
                    *child = new_child;
                }

                // 2. Check if this is a wrapper (1 child, and that child is a Leaf)
                if children.len() == 1 {
                    let (_key, child) = children.iter().next().unwrap();
                    if let ModuleNode::Leaf(_) = child {
                        return child.clone();
                    }
                }

                ModuleNode::Struct(children)
            }
            ModuleNode::Array(mut children) => {
                for child in children.iter_mut() {
                    let new_child = self.flatten_wrappers(child.clone());
                    *child = new_child;
                }
                ModuleNode::Array(children)
            }
            leaf => leaf,
        }
    }

    fn insert_into_tree(
        &self,
        node: &mut std::collections::BTreeMap<String, ModuleNode>,
        parts: &[&str],
        meta: LayerMeta,
    ) {
        if parts.is_empty() {
            return;
        }

        let key = parts[0];
        if parts.len() == 1 {
            let should_insert = if let Some(existing) = node.get(key) {
                if let ModuleNode::Struct(_) = existing {
                    // If we have a Struct but this new Leaf says it should be a Leaf (e.g. ParametrizedConv1d),
                    // we overwrite the Struct (pruning children).
                    self.should_be_leaf(&meta.module_type)
                } else {
                    false
                }
            } else {
                true
            };

            if should_insert {
                node.insert(key.to_string(), ModuleNode::Leaf(meta));
            }
        } else {
            let entry = node
                .entry(key.to_string())
                .or_insert_with(|| ModuleNode::Struct(std::collections::BTreeMap::new()));

            if let ModuleNode::Leaf(leaf_meta) = entry {
                if self.should_be_leaf(&leaf_meta.module_type) {
                    return; // Stop recursion for forced leaves
                }
                // Convert Leaf to Struct if we need to go deeper (Leaf was a container)
                *entry = ModuleNode::Struct(std::collections::BTreeMap::new());
            }

            if let ModuleNode::Struct(children) = entry {
                self.insert_into_tree(children, &parts[1..], meta);
            }
        }
    }

    fn compress_arrays(&self, node: ModuleNode) -> ModuleNode {
        match node {
            ModuleNode::Struct(mut children) => {
                // 1. Recurse first
                for (_, child) in children.iter_mut() {
                    let new_child = self.compress_arrays(child.clone());
                    *child = new_child;
                }

                // 2. Check for integer keys
                let mut indices = Vec::new();
                for key in children.keys() {
                    if let Ok(idx) = key.parse::<usize>() {
                        indices.push(idx);
                    } else {
                        return ModuleNode::Struct(children);
                    }
                }

                if indices.is_empty() {
                    return ModuleNode::Struct(children);
                }

                indices.sort();
                if indices[0] == 0 && indices.len() == (indices.last().unwrap() + 1) {
                    // Check homogeneity
                    let first_key = indices[0].to_string();
                    let first_node = &children[&first_key];

                    for i in 1..indices.len() {
                        let key = i.to_string();
                        let node = &children[&key];
                        if !self.are_nodes_compatible(first_node, node) {
                            return ModuleNode::Struct(children);
                        }
                    }

                    // It's a valid array
                    let mut array = Vec::new();
                    for i in 0..indices.len() {
                        array.push(children.remove(&i.to_string()).unwrap());
                    }
                    return ModuleNode::Array(array);
                }

                ModuleNode::Struct(children)
            }
            ModuleNode::Array(mut children) => {
                for child in children.iter_mut() {
                    let new_child = self.compress_arrays(child.clone());
                    *child = new_child;
                }
                ModuleNode::Array(children)
            }
            leaf => leaf,
        }
    }

    fn are_nodes_compatible(&self, a: &ModuleNode, b: &ModuleNode) -> bool {
        match (a, b) {
            (ModuleNode::Leaf(la), ModuleNode::Leaf(lb)) => la.module_type == lb.module_type,
            (ModuleNode::Struct(sa), ModuleNode::Struct(sb)) => {
                if sa.len() != sb.len() {
                    return false;
                }
                for (k, v_a) in sa {
                    if let Some(v_b) = sb.get(k) {
                        if !self.are_nodes_compatible(v_a, v_b) {
                            return false;
                        }
                    } else {
                        return false;
                    }
                }
                true
            }
            (ModuleNode::Array(aa), ModuleNode::Array(ab)) => {
                if aa.is_empty() || ab.is_empty() {
                    return true;
                }
                self.are_nodes_compatible(&aa[0], &ab[0])
            }
            _ => false,
        }
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
