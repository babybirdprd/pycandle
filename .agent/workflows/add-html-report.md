---
description: Add HTML report generation to PyCandle CLI
---

# Add HTML Report Generation

Generate a standalone HTML report showing module coverage and gaps.

## CLI Interface

```bash
pycandle report --manifest traces/manifest.json --out report.html
```

## Steps

### 1. Add Report command to main.rs

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing commands ...
    
    /// Generate an HTML coverage report
    Report {
        /// Path to the manifest JSON file
        #[arg(short, long)]
        manifest: PathBuf,
        
        /// Output HTML file path
        #[arg(short, long, default_value = "pycandle_report.html")]
        out: PathBuf,
    },
}
```

### 2. Create report.rs module

```rust
// src/report.rs
use crate::LayerMeta;
use std::collections::HashMap;

pub struct ReportGenerator {
    manifest: HashMap<String, LayerMeta>,
}

impl ReportGenerator {
    pub fn new(manifest: HashMap<String, LayerMeta>) -> Self {
        Self { manifest }
    }
    
    pub fn analyze(&self) -> ReportData {
        let mut supported = 0;
        let mut unsupported = 0;
        let mut gaps: HashMap<String, usize> = HashMap::new();
        
        for (name, meta) in &self.manifest {
            if self.is_supported(&meta.module_type) {
                supported += 1;
            } else {
                unsupported += 1;
                *gaps.entry(meta.module_type.clone()).or_default() += 1;
            }
        }
        
        ReportData { supported, unsupported, gaps, layers: self.manifest.clone() }
    }
    
    fn is_supported(&self, module_type: &str) -> bool {
        matches!(module_type, 
            "Linear" | "Conv1d" | "Conv2d" | "Embedding" | "LayerNorm"
        )
    }
    
    pub fn generate_html(&self, data: &ReportData) -> String {
        format!(r#"<!DOCTYPE html>
<html>
<head>
    <title>PyCandle Coverage Report</title>
    <style>
        body {{ font-family: system-ui; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        .dashboard {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; }}
        .card {{ background: #f5f5f5; padding: 20px; border-radius: 8px; }}
        .supported {{ color: #22c55e; }}
        .unsupported {{ color: #ef4444; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ text-align: left; padding: 8px; border-bottom: 1px solid #ddd; }}
    </style>
</head>
<body>
    <h1>PyCandle Coverage Report</h1>
    <div class="dashboard">
        <div class="card">
            <h2>Total Layers</h2>
            <p style="font-size: 48px;">{}</p>
        </div>
        <div class="card">
            <h2 class="supported">Supported</h2>
            <p style="font-size: 48px;">{}</p>
        </div>
        <div class="card">
            <h2 class="unsupported">Needs Implementation</h2>
            <p style="font-size: 48px;">{}</p>
        </div>
    </div>
    <h2>Gap Analysis</h2>
    <table>
        <tr><th>Module Type</th><th>Count</th><th>Status</th></tr>
        {}
    </table>
    <h2>All Layers</h2>
    <table>
        <tr><th>Layer Name</th><th>Type</th><th>Input Shape</th><th>Output Shape</th></tr>
        {}
    </table>
</body>
</html>"#,
            data.supported + data.unsupported,
            data.supported,
            data.unsupported,
            self.render_gaps_table(&data.gaps),
            self.render_layers_table(&data.layers),
        )
    }
}
```

### 3. Handle Report command in main

```rust
Commands::Report { manifest, out } => {
    let manifest_content = std::fs::read_to_string(&manifest)?;
    let manifest: HashMap<String, LayerMeta> = serde_json::from_str(&manifest_content)?;
    
    let generator = ReportGenerator::new(manifest);
    let data = generator.analyze();
    let html = generator.generate_html(&data);
    
    std::fs::write(&out, html)?;
    println!("📊 Report generated: {:?}", out);
}
```

## Testing

```bash
cargo run -- report --manifest chatterbox-repo/py_trace/s3gen_components_manifest.json --out report.html
start report.html  # Opens in browser on Windows
```
