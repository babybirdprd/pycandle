# PyCandle Meta-Trace Strategy

The **Meta-Trace Strategy** is the recommended SOP for porting large deep learning models (e.g., Transformers, LLMs) to Candle, especially on machines with limited RAM (e.g., 16GB RAM / 4GB available).

## The Problem: Paging File Errors
Loading weight files directly into PyTorch (e.g., 1.8GB + intermediate allocations) often exhausts system memory, leading to `OSError: The paging file is too small`.

## The Solution: Record Architecture on `meta`
Instead of loading weights into memory, we initialize the model on the PyTorch `meta` device. This records the **shapes**, **parameters**, and **computation graph** without allocating any actual weight buffers.

### Step 1: Initialize on Meta Device
Update your recording script to use the `meta` device.

```python
import torch
from py.spy import GoldenRecorder
from my_model import MyLargeModel

device = "meta"
with torch.device(device):
    model = MyLargeModel()
model.eval()

# Dummy inputs must also be on meta
dummy_input = torch.randn(1, 128, model.dim, device=device)

recorder = GoldenRecorder(output_dir="pycandle_trace")
recorder.record(model, dummy_input)
recorder.save("my_large_model_meta")
```

### Step 2: Generate Rust Code
Run the `pycandle codegen` as usual. The manifest contains everything needed to define the Rust structs.

```bash
cargo run -- codegen --manifest pycandle_trace/my_large_model_meta_manifest.json --out generated_model.rs
```

### Step 3: Stream-Extract Weights
Since you didn't record weights, your `trace.safetensors` will be empty or missing tensors. Use a specialized extraction script to map weights from the original checkpoint directly to the manifest-compatible keys.

```python
from safetensors.torch import load_file, save_file
# Load only what manifests says we need
# Rename keys to match manifest
# Save to a new, minimal weights file
```

## Benefits
- **Zero Memory Consumption**: Tracing happens in milliseconds with no RAM pressure.
- **Permanent Solution**: Works for models of any size (7B, 70B+) as long as the architecture fits in memory.
- **Power User DX**: Separates architecture capture from weight management.
