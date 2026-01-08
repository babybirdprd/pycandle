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

PyCandle has evolved into a high-fidelity transpilation framework. The following items track the transition from v0.1 to a production-grade v1.0.

### 🔄 DAG Resolver (torch.fx Tracing)
**Status:** Complete ✅

Handle non-sequential models with skip connections and branches:
- Use `torch.fx` to trace computation graphs automatically.
- Generate named variables based on FX graph nodes (e.g., `let x_conv1 = ...`).
- Automatic residual detection and mapping of functional ops (`add`, `cat`, `mul`).
- Mapping of `operator.getitem` to Candle `.i()` for complex slicing logic.

### 📐 Symbolic Shape Propagation
**Status:** Complete ✅

Generate `Config` structs instead of hardcoded dimensions to decouple model logic from specific input sizes:
- Automatic detection of `vocab_size`, `hidden_dim`, and `context_length`.
- Config-driven initialization in the generated Rust `load` functions.

### 📊 Visual Drift Analysis (Mechanistic Diagnostics)
**Status:** Complete ✅

Enhanced diagnostics for numerical drift using real-time verification data:
- **D3.js Coverage Report:** Standalone HTML report with MSE/Cosine Similarity charts.
- **Divergence Detection:** Automatic identification of the "Point of Failure" where math starts to drift.
- **TUI Dashboard:** Real-time terminal dashboard with 8x8 error heatmaps for instant visual feedback.

### 🎵 Audio-Specific Ops (pycandle-audio)
**Status:** Complete ✅

Bit-perfect PyTorch parity for audio preprocessing and specialized layers:
- **MelSpectrogram:** Full parity with `torchaudio` (including Slaney-scale area normalization).
- **STFT/iSTFT:** High-precision CPU-based transforms using `realfft`.
- **Specialized Layers:** Native support for `Snake` (BigVGAN), `CausalConv1d`, and `Mish`.

### 🔬 Interactive Debugger (Lock-Step)
**Status:** Complete ✅

Automated post-mortem tools for failed parity checks:
- **Snippet Generation:** Automatically saves `.safetensors` containing the failing Rust tensor and the Golden reference.
- **Python Debug Scripts:** Generates a ready-to-run `.py` script that loads the failure snippet into Matplotlib for visual histogram/heatmap comparison.

### 📦 Surgical Weight Management
**Status:** Complete ✅

Tools to handle the "Integration Gap" between PyTorch checkpoints and Rust structs:
- **Checkpoint Mapper:** Regex-based renaming engine to map PyTorch keys to Rust fields.
- **Meta-Extractor:** CLI tool to surgically extract only the weights used in a manifest, significantly reducing checkpoint size.

### 🧪 Automated Test Generation
**Status:** Planned

Eliminate manual test writing by generating the full Rust test harness:
- **Auto-Test CLI:** `pycandle gen-test` to generate `tests/parity.rs` automatically.
- **Data-Driven Harness:** The test will automatically load the input/output tensors from the recorded trace and run the verification loop.

### 🧩 Symbolic Ambiguity Hints
**Status:** In Progress 🛠️

Refining the symbolic propagator for complex models:
- **Hint System:** Allow users to provide a `hints.json` to resolve ambiguous dimensions (e.g., when `hidden_dim` and `context_length` are both 1024).
- **Dynamic Sequence Support:** Improved handling of variable-length audio/text sequences in the generated Rust code.

### 📉 Quantization Parity (GGUF/AWQ)
**Status:** Planned

Extending the verification engine to quantized models:
- **Quantization Drift Tracking:** Measure MSE drift introduced specifically by `Q4_0`, `Q8_0`, or `AWQ` compared to the `f32` Golden Record.
- **Parity-Aware Quantization:** Identify which specific layers are most sensitive to quantization to help guide mixed-precision strategies.

---

### Summary of the "Powerful" PyCandle Vision:
1.  **Python Spy:** Captures Graph (FX) + Config + Activations + Weights.
2.  **Transpiler:** Converts FX Graph to idiomatic Rust DAG (with residuals).
3.  **Verifiable Crate:** Generated code with `py_check!` macros that "lights up" green as you implement layers.
4.  **Diagnostics:** A visual report and Python debug scripts showing exactly where the "Math Leak" is happening.
---

## License

MIT
