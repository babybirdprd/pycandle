# PyCandle

**Automated PyTorch → Candle porting with layer-wise parity verification.**

PyCandle captures activation traces from PyTorch models to generate verified Rust Candle code. Acting as "Chrome DevTools" for neural networks, it provides full transparency into the internal state of complex models, ensuring seamless parity across Python and Rust.

## Quick Start

### 1. Initialize Project
Run `pycandle init` to set up your project. This creates a `.pycandle` directory with a managed virtual environment and necessary scripts.

```bash
pycandle init
```

Start the recording template:
```bash
# Edit the generated recorder to import your model
code .pycandle/scripts/recorder.py
```

### 2. Record Trace
Run the recorder to capture the model's architecture and weights (or just architecture if using meta device).

```bash
pycandle record --script .pycandle/scripts/recorder.py --name my_model
```

### 3. Generate Code
Generate the Rust implementation from the captured trace.

```bash
pycandle codegen --manifest .pycandle/traces/my_model_manifest.json --out src/model.rs
```

### 4. Parity Check
Run the generated parity tests to ensure the Rust implementation matches PyTorch exactly.

```bash
cargo test
```

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

### Advanced: Resolving Symbolic Ambiguity (Hints)

If your model has multiple dimensions with the same size (e.g., `hidden_dim=1024` and `context_length=1024`), the symbolic propagator might pick the wrong one. You can resolve this by passing `hints` to `recorder.save()`:

```python
recorder.save("my_model", hints={
    "vocab_size": 50257,
    "hidden_dim": 768,
    "context_length": 1024
})
```

The codegen will prioritize these hints when generating the `Config` struct and mapping layer dimensions.

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

## Extending PyCandle

Adding support for a new PyTorch module involves three main steps:

### 1. Implement the layer in Rust
Add the layer implementation to `crates/pycandle-core/src/layers/`. 
- **Activations**: Add to [activations.rs](file:///d:/pycandle/crates/pycandle-core/src/layers/activations.rs)
- **Normalization**: Add to [norm.rs](file:///d:/pycandle/crates/pycandle-core/src/layers/norm.rs)
- **Specialized**: Create a new file or add to [special.rs](file:///d:/pycandle/crates/pycandle-core/src/layers/special.rs)

### 2. Update Codegen Mapping
Register the mapping from the PyTorch module name to your Rust implementation:
- **Initialization**: Update `generate_init` in [renderer.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/renderer.rs).
- **Type Mapping**: Update `map_type` in [utils.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/utils.rs).
- **Verification**: Update `is_supported` in [utils.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/utils.rs).

### 3. Capture Metadata (Optional)
If your layer requires special configuration (like `eps`, `alpha`, or custom attributes), update `_get_module_config` in [py/spy.py](file:///d:/pycandle/py/spy.py).

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
**Status:** Complete ✅

Eliminate manual test writing by generating the full Rust test harness:
- **Auto-Test CLI:** `pycandle gen-test` to generate `tests/parity.rs` automatically.
- **Data-Driven Harness:** The test will automatically load the input/output tensors from the recorded trace and run the verification loop.

### 📐 Symbolic Ambiguity Hints
**Status:** Complete ✅

Refining the symbolic propagator for complex models:
- **Hint System:** Allow users to provide a `hints.json` to resolve ambiguous dimensions (e.g., when `hidden_dim` and `context_length` are both 1024).

### 📉 Quantization Parity (GGUF/AWQ)
**Status:** Complete ✅

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
### 🌐 Universal ONNX Transpilation
**Status:** Complete ✅

- **Bridge Strategy:** Automatically converts ONNX models to PyTorch in-memory, then traces them with `torch.fx` to generate idiomatic Rust.
- **CLI Integration:** `pycandle onnx-convert --onnx model.onnx --name my_model` handles the conversion pipeline automatically.
- **Dynamic Shape Inference:** Automatically detects input dimensions from the ONNX graph definition, defaulting dynamic axes to `1`.

### 🛡️ Production Refactor (v1.0 God-Object Reduction)
**Status:** Complete ✅

- **Modular Layers**: Split the giant `layers.rs` into specialized sub-modules ([activations.rs](file:///d:/pycandle/crates/pycandle-core/src/layers/activations.rs), [norm.rs](file:///d:/pycandle/crates/pycandle-core/src/layers/norm.rs), etc.).
- **Codegen Separation**: Modularized the [codegen](file:///d:/pycandle/crates/pycandle-core/src/codegen/mod.rs) module into [types.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/types.rs), [ops.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/ops.rs), and [renderer.rs](file:///d:/pycandle/crates/pycandle-core/src/codegen/renderer.rs) for better maintainability.

---

## License

MIT
