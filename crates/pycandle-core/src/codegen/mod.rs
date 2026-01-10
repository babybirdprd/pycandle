use crate::LayerMeta;
use std::collections::HashMap;

pub mod analyzer;
pub mod config;
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
