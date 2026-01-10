use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// Internal Types for Codegen
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReturnType {
    Tensor,
    Tuple,
    Vec,
    Primitive,
    FInfo, // For torch.finfo objects
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

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct LayerInfo {
    pub name: String,
    pub module_type: String,
    pub supported: bool,
    pub input_shapes: Vec<Vec<usize>>,
    pub output_shapes: Vec<Vec<usize>>,
}

#[derive(Deserialize, Debug, Clone)]
pub struct GraphNode {
    pub name: String,
    pub op: String,
    pub target: String,
    pub args: Vec<serde_json::Value>,
    #[serde(default)]
    pub kwargs: HashMap<String, serde_json::Value>,
    pub module_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SymbolicConfig {
    pub dims: HashMap<String, usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModuleNode {
    Leaf(crate::LayerMeta),
    Struct(std::collections::BTreeMap<String, ModuleNode>),
    Array(Vec<ModuleNode>),
}
