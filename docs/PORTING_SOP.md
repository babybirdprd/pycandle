# PyCandle Porting SOP

This document outlines the Standard Operating Procedure (SOP) for porting PyTorch models to Rust/Candle using the PyCandle framework.

## 1. Project Setup
Initialize a new porting project. This generates the necessary boilerplate for recording and parity testing.

```bash
# In your workspace root
pycandle init --name <RunName>
```

**What this does:**
- Creates `recorder.py`: A standard PyTorch recording script tailored for your model.
- Creates `tests/main.rs`: A standard integration test harness for parity verification.

## 2. Model Wrapping (Python)
Edit the generated `recorder.py` to import and instantiate your specific PyTorch model.

```python
# recorder.py
from my_model_source import MyModel  # <--- Import your model

model = MyModel(...)
input_tensor = torch.randn(...)
```

## 3. Recording Traces
Run the recorder to capture the Golden Trace (activations, weights, and config).

```bash
# Uses uv under the hood to manage dependencies
pycandle record --script recorder.py --name <RunName>
```

*Output:* `traces/<RunName>/` containing `.safetensors` and `_manifest.json`.

## 4. Codegen & Analysis
Analyze the trace to see what layers are supported and generate the Rust code.

```bash
# Analyze first
pycandle codegen --manifest traces/<RunName>/_manifest.json --analyze-only

# Generate Rust code
pycandle codegen --manifest traces/<RunName>/_manifest.json --out crates/my-model/src/model.rs --model MyModel
```

## 5. Parity Verification
Run the generated test harness to verify that your Rust implementation matches the PyTorch golden record bit-for-bit.

```bash
cargo test
```

## 6. Iterative Refinement
If `cargo test` fails or parity is low:
1.  Use `pycandle dashboard` to visualize the error.
2.  Fix the Rust implementation.
3.  Re-run `cargo test`.
