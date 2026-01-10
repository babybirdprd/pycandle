use super::types::{AnalysisResult, GapInfo, LayerInfo};
use std::collections::HashMap;

impl super::Codegen {
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
}
