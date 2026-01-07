---
description: Add JSON output mode to PyCandle CLI for agent integration
---

# Add JSON Output Mode

Add `--json` flag to all CLI commands for machine-readable output.

## CLI Interface

```bash
# Analyze gaps without generating code
pycandle codegen --manifest m.json --analyze-only --json

# Output:
# {"supported":45,"unsupported":12,"gaps":[{"type":"LSTM","count":1},...]}
```

## Steps

### 1. Add global JSON flag

```rust
#[derive(Parser)]
struct Cli {
    /// Output in JSON format for agent consumption
    #[arg(long, global = true)]
    json: bool,
    
    #[command(subcommand)]
    command: Commands,
}
```

### 2. Add analyze-only flag to Codegen

```rust
Commands::Codegen {
    manifest: PathBuf,
    out: PathBuf,
    model: String,
    
    /// Analyze without generating code
    #[arg(long)]
    analyze_only: bool,
}
```

### 3. Create JSON output structs

```rust
#[derive(Serialize)]
struct AnalysisResult {
    supported: usize,
    unsupported: usize,
    total: usize,
    coverage_percent: f32,
    gaps: Vec<GapInfo>,
    layers: Vec<LayerInfo>,
}

#[derive(Serialize)]
struct GapInfo {
    module_type: String,
    count: usize,
    suggestion: String,
}

#[derive(Serialize)]
struct LayerInfo {
    name: String,
    module_type: String,
    supported: bool,
    input_shapes: Vec<Vec<i64>>,
    output_shapes: Vec<Vec<i64>>,
}
```

### 4. Implement analysis in codegen command

```rust
Commands::Codegen { manifest, out, model, analyze_only } => {
    let manifest_content = std::fs::read_to_string(&manifest)?;
    let manifest: HashMap<String, LayerMeta> = serde_json::from_str(&manifest_content)?;
    
    let codegen = Codegen::new(manifest.clone());
    let analysis = codegen.analyze();
    
    if cli.json {
        println!("{}", serde_json::to_string_pretty(&analysis)?);
    } else if analyze_only {
        println!("📊 Analysis:");
        println!("  Supported: {}/{}", analysis.supported, analysis.total);
        println!("  Gaps: {:?}", analysis.gaps);
    } else {
        let code = codegen.generate_model_rs(&model);
        std::fs::write(&out, code)?;
        println!("✅ Code generated: {:?}", out);
    }
}
```

### 5. Add analyze method to Codegen

```rust
impl Codegen {
    pub fn analyze(&self) -> AnalysisResult {
        let mut supported = 0;
        let mut unsupported = 0;
        let mut gap_counts: HashMap<String, usize> = HashMap::new();
        let mut layers = Vec::new();
        
        for (name, meta) in &self.manifest {
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
        
        let gaps: Vec<GapInfo> = gap_counts.into_iter()
            .map(|(t, c)| GapInfo {
                module_type: t.clone(),
                count: c,
                suggestion: self.get_suggestion(&t),
            })
            .collect();
        
        AnalysisResult {
            supported,
            unsupported,
            total: supported + unsupported,
            coverage_percent: (supported as f32 / (supported + unsupported) as f32) * 100.0,
            gaps,
            layers,
        }
    }
    
    fn get_suggestion(&self, module_type: &str) -> String {
        match module_type {
            "LSTM" => "Use /add-lstm workflow".to_string(),
            "BatchNorm1d" | "BatchNorm2d" => "Use /add-batchnorm workflow".to_string(),
            "Snake" | "ELU" => "Use /add-activations workflow".to_string(),
            _ => format!("Implement {} manually", module_type),
        }
    }
}
```

## Agent Usage Example

```bash
# Get gaps as JSON
GAPS=$(pycandle codegen --manifest m.json --analyze-only --json)

# Parse with jq
echo $GAPS | jq '.gaps[] | select(.count > 5)'
```

## Testing

```bash
cargo run -- codegen --manifest chatterbox-repo/py_trace/s3gen_components_manifest.json --analyze-only --json
```
