# PyCandle

**Automated PyTorch → Candle porting with layer-wise parity verification.**

PyCandle captures activation traces from PyTorch models to generate verified Rust Candle code. Acting as "Chrome DevTools" for neural networks, it provides full transparency into the internal state of complex models, ensuring seamless parity across Python and Rust.

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
cargo run -p pycandle -- codegen \
    --manifest traces/my_model_manifest.json \
    --out generated_my_model.rs \
    --model MyModel
```

### 3. Use generated code with parity checking

```rust
use pycandle_core::{PyChecker, py_check};

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
use pycandle_core::{PyChecker, py_check};

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
pub struct Config {
    pub vocab_size: usize,
    pub hidden_dim: usize,
}

pub struct MyModel {
    pub linear1: Linear,
    pub linear2: Linear,
    pub checker: Option<PyChecker>,
}

impl MyModel {
    pub fn load(cfg: Config, vb: VarBuilder, checker: Option<PyChecker>) -> Result<Self> {
        let linear1 = candle_nn::linear(cfg.hidden_dim, 256, vb.pp("linear1"))?;
        let linear2 = candle_nn::linear(256, cfg.vocab_size, vb.pp("linear2"))?;
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
| `nn.Linear` | `candle_nn::linear` | ✅ Auto (with smart transpose) |
| `nn.Conv1d` | `candle_nn::conv1d` | ✅ Auto |
| `nn.Embedding` | `candle_nn::embedding` | ✅ Auto |
| `nn.LayerNorm` | `candle_nn::layer_norm` | ✅ Auto |
| `nn.BatchNorm1d` | `BatchNorm1d` | ✅ Auto |
| `nn.BatchNorm2d` | `BatchNorm2d` | ✅ Auto |
| `nn.LSTM` | `LSTM` | ✅ Auto |
| `nn.ReLU/GELU/Sigmoid/Tanh` | Activations | ✅ Auto |
| `nn.ELU/LeakyReLU` | Parameterized activations | ✅ Auto |
| `Snake` (BigVGAN) | `Snake` | ✅ Auto |
| Custom modules | - | ⚠️ TODO marker |

## Workspace Structure

```
pycandle/
├── Cargo.toml              # Workspace root
├── crates/
│   ├── pycandle/           # CLI binary
│   │   └── src/main.rs
│   ├── pycandle-core/      # Library (PyChecker, layers, codegen)
│   │   └── src/
│   │       ├── lib.rs
│   │       ├── checker.rs
│   │       ├── layers.rs
│   │       └── codegen/
│   └── pycandle-audio/     # Audio ops (STFT, padding)
│       └── src/lib.rs
└── py/
    └── spy.py              # GoldenRecorder
```

**Using as a library:**
```toml
[dependencies]
pycandle-core = { git = "https://github.com/user/pycandle" }
# Optional audio support:
pycandle-audio = { git = "https://github.com/user/pycandle" }
```

---

## Roadmap

These features are planned or currently in development to move PyCandle from a utility to a production-grade transpilation framework:

### 🔄 DAG Resolver (torch.fx Tracing)
**Status:** Complete ✅

Handle non-sequential models with skip connections and branches:
- Use `torch.fx` to trace computation graphs automatically via `recorder.record(model, x, trace_fx=True)`
- Generate named variables based on FX graph nodes instead of sequential `x = layer(x)`
- Automatic residual detection: `let out = (&x_bn2 + &xs)?;`
- Support for tensor concatenation, branching, and common tensor methods (`view`, `reshape`, `flatten`, etc.)

### 📐 Symbolic Shape Propagation
**Status:** Complete ✅

Generate Config structs instead of hardcoded dimensions. This decouples the model logic from specific input sizes:
```rust
pub struct Config {
    pub context_length: usize, // 1024
    pub hidden_dim: usize,     // 768
    pub vocab_size: usize,     // 50257
}
```

### 📊 Visual Drift Analysis (Mechanistic Diagnostics)
**Status:** Complete ✅

Enhanced diagnostics for numerical drift using real verification data:
- **Live Data Ingestion:** Update `PyChecker` to export `verification_results.json` which populates the D3.js report.
- **MSE Heatmap:** Visual representation of "Math Leakage" across layers.
- **Point of Divergence:** Automatically highlight the exact layer where `Cosine Similarity` drops below `0.99`.

### 🎵 Audio-Specific Ops (pycandle-audio)
**Status:** Complete ✅

PyTorch-parity audio operations:
- [x] Reflect/Replicate/Constant padding (`pad_1d`)
- [x] Hann window generation
- [x] STFT/iSTFT (CPU-based using `realfft`)
- [x] Bit-perfect `MelSpectrogram` implementation matching `torchaudio.transforms` (including Slaney-scale parity).

### 🛠️ Advanced FX Logic Mapping
**Status:** Complete ✅

Expanding the DAG resolver to handle complex Pythonic tensor manipulation:
- **Slicing/Indexing:** Map PyTorch `x[:, :10]` (operator.getitem) to Candle `.narrow()` or `.i()`.
- **Tuple/Multi-Output:** Support for modules that return multiple tensors (e.g., Attention weights or RNN hidden states).
- **Chunk/Split:** Native mapping for `torch.chunk` and `torch.split` used in GLU activations.

### 📦 Surgical Weight Management
**Status:** Complete ✅

Tools to handle the "Integration Gap" between PyTorch checkpoints and Rust structs:
- **Checkpoint Mapper:** JSON-based renaming engine to map PyTorch keys (`encoder.block.0`) to Rust field names (`h.0`) without code changes.
- **Meta-Extractor:** CLI tool to read a multi-gigabyte PyTorch file and extract *only* the tensors defined in the `manifest.json`, renaming them on the fly for Candle.

### 🔬 Interactive Debugger (Lock-Step)
**Status:** Complete ✅

When a `py_check!` fails in Rust:
- Save a `.safetensors` snippet containing the erroneous Rust tensor and the Golden reference.
- Generate a Python comparison script for side-by-side inspection in a Jupyter notebook.
- ASCII terminal "Mini-Heatmap" for quick debugging in CI environments.

### ⚡ Minimal Developer Code
**Status:** Planned

One-command project setup:
- `pycandle init` - Detect project structure and generate a boilerplate recording script.
- Auto-detect model entry points from `pyproject.toml`.
- Generate ready-to-run verification binaries automatically.

---

### Summary of the "Powerful" PyCandle Vision:
1.  **Python Spy:** Captures Graph (FX) + Config + Activations + Weights.
2.  **Transpiler:** Converts FX Graph to idiomatic Rust DAG (with residuals).
3.  **Verifiable Crate:** Generated code with `py_check!` macros that "lights up" green as you implement layers.
4.  **Diagnostics:** A visual report showing exactly where the "Math Leak" is happening.

---

## License

MIT
