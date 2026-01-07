# PyCandle

**Automated PyTorch → Candle porting with layer-wise parity verification.**

PyCandle records activation traces from PyTorch models and generates Rust Candle code with embedded verification hooks.

## Quick Start

### 1. Record a PyTorch model

```python
# your_model_script.py
import sys
sys.path.insert(0, "path/to/pycandle/py")

import torch
from spy import GoldenRecorder

# Your PyTorch model
model = MyModel()
model.eval()

# Create dummy input
x = torch.randn(1, 128)

# Record
recorder = GoldenRecorder(output_dir="traces")
recorder.record(model, x)
recorder.save("my_model")
```

### 2. Generate Candle code

```bash
cargo run -- codegen \
    --manifest traces/my_model_manifest.json \
    --out generated_my_model.rs \
    --model MyModel
```

### 3. Use generated code with parity checking

```rust
use pycandle::{PyChecker, py_check};

// Load golden records for verification
let checker = PyChecker::load("my_model", "traces/", &device)?;

// Use the generated model
let model = MyModel::load(vb, Some(checker))?;
let output = model.forward(&input)?;  // py_check! runs at each layer
```

## CLI Commands

### `pycandle record`
Run a Python script that uses `GoldenRecorder`.

```bash
pycandle record --script model_script.py --name my_model --out traces/
```

### `pycandle codegen`
Generate Candle Rust code from a manifest.

```bash
pycandle codegen --manifest traces/manifest.json --out generated.rs --model ModelName
```

**Flags:**
- `--analyze-only` - Show analysis without generating code
- `--json` - Output as JSON (works with all commands)

**Analysis mode example:**
```bash
# Human-readable analysis
pycandle codegen --manifest m.json --out NUL --analyze-only

# JSON output for scripting
pycandle codegen --manifest m.json --out NUL --analyze-only --json
```

### `pycandle todos`
Extract and manage TODO markers in generated code.

```bash
# List all TODOs with suggestions
pycandle todos --file generated_model.rs

# JSON output
pycandle todos --file generated_model.rs --json

# Check mode (exit code 1 if TODOs remain)
pycandle todos --file generated_model.rs --check
```

**Agent workflow example:**
```bash
# 1. Generate code
pycandle codegen --manifest m.json --out model.rs --model MyModel

# 2. Check for TODOs
if ! pycandle todos --file model.rs --check; then
    # 3. Get gaps as JSON and implement
    pycandle todos --file model.rs --json | jq '.by_type'
fi
```

## Python Spy API

```python
from spy import GoldenRecorder

recorder = GoldenRecorder(output_dir="traces")
recorder.record(model, *inputs, **kwargs)  # Runs forward pass with hooks
recorder.save("project_name")  # Saves .safetensors + _manifest.json
```

**Output files:**
- `{name}_trace.safetensors` - Activation tensors for each layer
- `{name}_manifest.json` - Module metadata (types, shapes, configs)

## Rust Verification API

```rust
use pycandle::{PyChecker, py_check};

// Load checker
let checker = PyChecker::load("model_name", "traces/", &device)?;

// Verify a tensor against golden record
let result = checker.verify("layer_name", &tensor)?;
println!("MSE: {}", result.mse);

// Or use the macro (embedded in generated code)
py_check!(checker, "layer_name", &tensor);
```

## Generated Code Structure

```rust
pub struct MyModel {
    pub linear1: Linear,
    pub linear2: Linear,
    pub checker: Option<PyChecker>,
}

impl MyModel {
    pub fn load(vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let linear1 = candle_nn::linear(128, 256, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(256, 10, vb.pp("linear2"))?;
        Ok(Self { linear1, linear2, checker })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let mut x = xs.clone();
        x = self.linear1.forward(&x)?;
        py_check!(self.checker, "linear1", &x);
        x = self.linear2.forward(&x)?;
        py_check!(self.checker, "linear2", &x);
        Ok(x)
    }
}
```

## Supported Module Types

| PyTorch | Candle | Status |
|---------|--------|--------|
| `nn.Linear` | `candle_nn::linear` | ✅ Auto |
| `nn.Conv1d` | `candle_nn::conv1d` | ✅ Auto |
| `nn.Embedding` | `candle_nn::embedding` | ✅ Auto |
| `nn.LayerNorm` | `candle_nn::layer_norm` | ✅ Auto |
| `nn.BatchNorm1d` | `BatchNorm1d` | ✅ Auto |
| `nn.BatchNorm2d` | `BatchNorm2d` | ✅ Auto |
| `nn.LSTM` | `LSTM` | ✅ Auto |
| Custom modules | - | ⚠️ TODO marker |

## Project Structure

```
pycandle/
├── src/
│   ├── main.rs          # CLI entry point
│   ├── lib.rs           # PyChecker, py_check! macro, layer implementations
│   ├── todos.rs         # TODO extraction from generated code
│   └── codegen/
│       ├── mod.rs       # Manifest → Rust code generator
│       └── gpt2.rs      # GPT2-specific helpers
├── py/
│   └── spy.py           # GoldenRecorder
└── Cargo.toml
```

## License

MIT
