use anyhow::{Context, Result};
use candle_core::{Shape, Tensor};
use candle_nn::VarBuilder;
use regex::Regex;
use std::collections::{HashMap, HashSet};

/// Auto-casts weights if they don't match the VarBuilder's dtype.
/// This prevents panics when loading f16 weights into an f32 model or vice-versa.
pub fn get_cast(
    vb: &VarBuilder,
    shape: impl Into<Shape>,
    name: &str,
) -> candle_core::Result<Tensor> {
    let tensor = vb.get(shape, name)?;
    if tensor.dtype() != vb.dtype() {
        tensor.to_dtype(vb.dtype())
    } else {
        Ok(tensor)
    }
}

/// A renaming engine for tensor keys using Regex patterns.
pub struct WeightMapper {
    mappings: Vec<(Regex, String)>,
}

impl WeightMapper {
    /// Create a mapper from a JSON string containing pattern -> replacement mappings.
    pub fn from_json(json: &str) -> Result<Self> {
        let raw: HashMap<String, String> = serde_json::from_str(json)?;
        let mut mappings = Vec::new();
        // Sort keys to ensure deterministic order (important for overlapping regexes)
        let mut keys: Vec<_> = raw.keys().collect();
        keys.sort();

        for pattern in keys {
            let replacement = raw.get(pattern).unwrap();
            let re = Regex::new(pattern)
                .with_context(|| format!("Invalid regex pattern: {}", pattern))?;
            mappings.push((re, replacement.clone()));
        }
        Ok(Self { mappings })
    }

    /// Map a key using all registered patterns.
    pub fn map_key(&self, key: &str) -> String {
        let mut current = key.to_string();
        for (re, replacement) in &self.mappings {
            current = re.replace_all(&current, replacement.as_str()).to_string();
        }
        current
    }
}

/// Identifies which weights are actually used in a model based on its manifest.
pub struct WeightExtractor {
    active_params: HashSet<String>,
}

impl WeightExtractor {
    /// Create an extractor from a manifest JSON.
    pub fn from_manifest(manifest_json: &str) -> Result<Self> {
        let manifest: HashMap<String, serde_json::Value> = serde_json::from_str(manifest_json)?;
        let mut active_params = HashSet::new();

        for (module_name, meta) in manifest {
            if module_name.starts_with('_') {
                continue;
            }

            if let Some(params) = meta.get("parameters").and_then(|p| p.as_array()) {
                for p in params {
                    if let Some(p_name) = p.as_str() {
                        // PyTorch module parameters are accessed as module_name.weight, etc.
                        active_params.insert(format!("{}.{}", module_name, p_name));
                    }
                }
            }
        }

        Ok(Self { active_params })
    }

    /// Check if a given tensor key is used in the manifest.
    pub fn should_keep(&self, key: &str) -> bool {
        self.active_params.contains(key)
    }

    /// Return the set of all active parameter keys.
    pub fn active_keys(&self) -> &HashSet<String> {
        &self.active_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mapper_basic() {
        let json = r#"{"encoder\\.layers\\.(\\d+)\\.": "h.$1."}"#;
        let mapper = WeightMapper::from_json(json).unwrap();
        assert_eq!(mapper.map_key("encoder.layers.0.weight"), "h.0.weight");
        assert_eq!(mapper.map_key("encoder.layers.15.bias"), "h.15.bias");
    }

    #[test]
    fn test_extractor_basic() {
        let manifest = r#"{
            "encoder.layers.0": {
                "parameters": ["weight", "bias"]
            },
            "decoder": {
                "parameters": ["weight"]
            },
            "_internal": { "parameters": ["unused"] }
        }"#;
        let extractor = WeightExtractor::from_manifest(manifest).unwrap();
        assert!(extractor.should_keep("encoder.layers.0.weight"));
        assert!(extractor.should_keep("encoder.layers.0.bias"));
        assert!(extractor.should_keep("decoder.weight"));
        assert!(!extractor.should_keep("encoder.layers.0.other"));
        assert!(!extractor.should_keep("_internal.unused"));
    }
}
