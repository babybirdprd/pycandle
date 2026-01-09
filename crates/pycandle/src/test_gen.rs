use anyhow::{Context, Result};
use pycandle_core::LayerMeta;
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

pub struct TestGenerator {
    model_name: String,
    crate_name: String,
    manifest: HashMap<String, LayerMeta>,
    project_name: String,
}

impl TestGenerator {
    pub fn new(model_name: String, manifest_path: PathBuf) -> Result<Self> {
        let manifest_content = fs::read_to_string(&manifest_path)
            .with_context(|| format!("Failed to read manifest at {:?}", manifest_path))?;

        let full_manifest: HashMap<String, serde_json::Value> =
            serde_json::from_str(&manifest_content).context("Failed to parse manifest JSON")?;

        let manifest: HashMap<String, LayerMeta> = full_manifest
            .into_iter()
            .filter(|(k, _)| !k.starts_with('_'))
            .map(|(k, v)| {
                let meta: LayerMeta = serde_json::from_value(v)
                    .with_context(|| format!("Failed to parse LayerMeta for {}", k))?;
                Ok((k, meta))
            })
            .collect::<Result<_>>()?;

        let stem = manifest_path.file_stem().unwrap().to_str().unwrap();
        let project_name = stem.replace("_manifest", "");

        Ok(Self {
            model_name,
            crate_name: Self::detect_crate_name().unwrap_or_else(|_| "my_project".to_string()),
            manifest,
            project_name,
        })
    }

    fn detect_crate_name() -> Result<String> {
        let content = fs::read_to_string("Cargo.toml").context("Failed to read Cargo.toml")?;
        let toml: toml::Value = toml::from_str(&content).context("Failed to parse Cargo.toml")?;

        if let Some(package) = toml.get("package") {
            if let Some(name) = package.get("name") {
                return Ok(name.as_str().unwrap_or("my_project").replace("-", "_"));
            }
        }

        Ok("my_project".to_string())
    }

    pub fn generate_test_file(&self) -> String {
        let input_count = self.count_model_inputs();
        let inputs_code = self.generate_inputs_loading_code(input_count);
        let forward_call = self.generate_forward_call(input_count);

        format!(
            r#"#[cfg(test)]
mod tests {{
    use super::*;
    use candle_core::{{Device, Tensor}};
    use pycandle_core::{{PyChecker, VerificationMode}};
    use {}::{};
    use anyhow::Result;

    #[test]
    fn test_parity() -> Result<()> {{
        // 1. Setup Device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Running on device: {{:?}}", device);

        // 2. Load Checker and Golden Trace
        // Assumes the trace directory is in the current project root
        let checker = PyChecker::load("{}", "pycandle_trace", &device)?
            .with_mode(VerificationMode::Strict);
        println!("Loaded checker with trace: {{}}", checker.name);

        // 3. Load Model
        // We use zeros VB as a placeholder; in a real parity test, 
        // you might want to load weights using pycandle weight tools.
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = {}::load(vb, Some(checker.clone()))?;

        // 4. Load Inputs from Trace
        let trace_path = format!("pycandle_trace/{}_trace.safetensors", "{}");
        let tensors = candle_core::safetensors::load(&trace_path, &device)?;
        
{}

        // 5. Run Forward Pass & Verify
{}
        println!("✅ Parity test passed for {}!");

        Ok(())
    }}
}}
"#,
            self.crate_name,
            self.model_name,
            self.project_name,
            self.model_name,
            self.project_name,
            self.project_name,
            inputs_code,
            forward_call,
            self.model_name
        )
    }

    fn count_model_inputs(&self) -> usize {
        // Simple heuristic: check manifest for model_input.N keys or count input shapes of top level
        // Actually, we can just look at what's in the trace if we had it,
        // but from manifest we rely on the fact that GoldenRecorder saves 'model_input.0'
        // For now, let's assume we can find up to 10.
        // A better way: the GoldenRecorder knows.
        // Let's just check for the existence of "model_input.0" in manifest if it was recorded there?
        // Actually GoldenRecorder saves them to records, but not necessarily manifest unless they are layers.
        // But we know 'model_input.0' is the standard naming.
        // Let's just default to 1 for now if we can't find more, or maybe look at the first layer's input count.
        1
    }

    fn generate_inputs_loading_code(&self, count: usize) -> String {
        let mut code = String::new();
        for i in 0..count {
            code.push_str(&format!(
                "        let x{} = tensors.get(\"model_input.{}\").context(\"Missing model_input.{}\")?.clone();\n",
                i, i, i
            ));
        }
        code
    }

    fn generate_forward_call(&self, count: usize) -> String {
        let args = (0..count)
            .map(|i| format!("&x{}", i))
            .collect::<Vec<_>>()
            .join(", ");

        // We need to know if the model returns a single tensor or multiple
        // For now, assume single.
        let output_layer = self.detect_output_layer();
        format!(
            r#"        let output = model.forward({})?;
        checker.verify("{}", &output)?;"#,
            args, output_layer
        )
    }

    fn detect_output_layer(&self) -> String {
        // Heuristic: The layer that isn't used as an input to any other layer,
        // or the last one alphabetically if it looks like a sequence.
        // If we have FX graph, it's the last node.
        // For now, return a placeholder or try to find the "last" one.
        if let Some(last) = self.manifest.keys().max() {
            last.clone()
        } else {
            "TODO_output_layer".to_string()
        }
    }
}
