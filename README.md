# 🕯️ PyCandle

**The "Chrome DevTools" for Neural Networks.**
*Automated, verified, production-grade PyTorch → Candle (Rust) transpiler.*

PyCandle is not just a porting script. It is a **compiler** that turns PyTorch execution graphs (`torch.fx`) into idiomatic, high-performance Rust code. It handles the math, the shapes, the weights, and the verification so you don't have to.

[![Rust](https://img.shields.io/badge/built_with-Rust-red)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-FX-orange)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Production_Ready-green)]()

## ⚡ Superpowers (Why use this?)

### 🧠 1. The "Einsum" Solver
PyCandle includes a symbolic math parser that translates complex `torch.einsum("bhld,hdm->bhlm")` equations into exact sequences of `permute` -> `matmul` -> `reshape` in Rust.

### 🚀 2. "Flash Attention" Auto-Switching
Detects `F.scaled_dot_product_attention` in your Python graph and generates Rust code that automatically switches between:
*   **Metal/Cuda:** `candle_flash_attn` kernels for SOTA speed.
*   **CPU/Fallback:** Exact mathematical implementation for compatibility.

### 🛡️ 3. Smart Broadcasting & Shape Safety
Candle is strict about shapes; PyTorch is loose. PyCandle generates **Smart Wrappers** around math operations (`pycandle::ops::add`) that automatically handle rank mismatches and broadcasting, eliminating 90% of runtime panics.

### 🎵 4. Audio "First-Class" Support
Full parity with `torchaudio`:
*   **STFT/iSTFT:** CPU-based `realfft` implementation matches PyTorch exactly.
*   **MelSpectrogram:** Includes Slaney-scale and area-normalization logic.
*   **Vocoder Ops:** Native support for `Snake`, `Mish`, and `ReflectPadding`.

### 🔬 5. Interactive "Lock-Step" Debugger
When a layer fails parity verification:
1.  PyCandle saves a `.safetensors` snippet of the failing Rust tensor vs the Golden Python tensor.
2.  It generates a `debug_layer.py` script.
3.  You run the script to see a **Visual Heatmap** of exactly where the math diverged.

---

## 🛠️ Workflow

### 1. Scaffold a New Project
Don't waste time setting up `Cargo.toml` or directory structures.
```bash
pycandle new my_llm --model GPT2
cd my_llm
pycandle init
```

### 2. Record the Python Model
PyCandle uses `torch.fx` to trace the execution graph, capturing not just layers, but logic (branches, loops, residuals).
```bash
# Edit .pycandle/scripts/recorder.py to load your model
pycandle record --script .pycandle/scripts/recorder.py --name my_llm
```

### 3. Compile to Rust
Transform the FX graph into idiomatic Rust.
```bash
# Generates src/lib.rs with structs, configs, and weight loading
pycandle codegen --manifest pycandle_trace/my_llm_manifest.json --out src/lib.rs --stateful
```

### 4. Verify Parity
Run the auto-generated test harness. It loads the Python activations and compares them against Rust layer-by-layer.
```bash
cargo test --release
```

### 5. Visualize
Launch the TUI Dashboard to see real-time pass/fail rates and drift heatmaps.
```bash
pycandle dashboard
```

---

## 📦 Advanced Features

### Surgical Weight Extraction
Don't ship 10GB checkpoint files if you only use 2GB of weights.
```bash
pycandle weights extract \
    --checkpoint big_model.bin \
    --manifest manifest.json \
    --out stripped_model.safetensors
```

### ONNX Bridge
Have an ONNX model? PyCandle can ingest it by converting it to an in-memory PyTorch graph first.
```bash
pycandle onnx-convert --onnx model.onnx --name my_model
```

### Benchmark Generation
Prove the speedup. Automatically generate `criterion` benchmarks using the shapes captured during the trace.
```bash
pycandle gen-bench --manifest manifest.json --out benches/inference.rs
```

---

## 🧩 Supported Layers & Ops

| Category | Modules |
|----------|---------|
| **Transformers** | Linear, Embedding, LayerNorm, RMSNorm, MultiHeadAttention (SDPA) |
| **Vision/Audio** | Conv1d, Conv2d, ConvTranspose, BatchNorm, GroupNorm |
| **Recurrent** | LSTM, GRU |
| **Activations** | ReLU, GELU, SiLU, Mish, Snake, Tanh, Sigmoid, LeakyReLU |
| **Logic** | Cat, Stack, Chunk, Split, Indexing (`[:, 1]`), Permute, Reshape, View |

---

## License

MIT
