# PyCandle Meta-Trace Strategy

The **Meta-Trace Strategy** is the recommended SOP for porting large deep learning models (e.g., Transformers, LLMs) to Candle, especially on machines with limited RAM (e.g., 16GB RAM / 4GB available).

## The Problem: Paging File Errors
Loading weight files directly into PyTorch (e.g., 1.8GB + intermediate allocations) often exhausts system memory, leading to `OSError: The paging file is too small`.

## The Solution: Record Architecture on `meta`
Instead of loading weights into memory, we initialize the model on the PyTorch `meta` device. This records the **shapes**, **parameters**, and **computation graph** without allocating any actual weight buffers.

### Step 0: Initialize Project
Ensure you have initialized your project with the standard `.pycandle` structure. This sets up the managed virtual environment and scripts.

```bash
pycandle init
```

### Step 1: Initialize on Meta Device
Edit your `.pycandle/scripts/recorder.py` to use the `meta` device.

```python
import torch
from spy import GoldenRecorder
from my_model import MyLargeModel

# 1. Use meta device for zero-memory allocation
device = "meta"
with torch.device(device):
    model = MyLargeModel()
model.eval()

# 2. Dummy inputs must also be on meta
dummy_input = torch.randn(1, 128, model.dim, device=device)

# 3. Record
recorder = GoldenRecorder(output_dir="../traces")
recorder.record(model, dummy_input)
recorder.save("my_large_model_meta")
```

Run the recorder using the CLI:
```bash
pycandle record
```

### Step 2: Generate Rust Code
Run the `codegen` command. The manifest contains everything needed to define the Rust structs.

```bash
pycandle codegen --manifest .pycandle/traces/my_large_model_meta_manifest.json --out src/model.rs
```

### Step 3: Surgically Extract Weights (OOM Safe)
Since you didn't record weights, your `trace.safetensors` will be empty. use the `weight_extractor` script to map weights from the original checkpoint directly to the manifest-compatible keys without loading the entire model into RAM.

**Tip:** If your source checkpoint is a pickle file (`.bin`/`.pt`), convert it to `.safetensors` first if possible, as `.safetensors` supports lazy loading (mmap), drastically reducing memory usage.

```bash
# If you have a manifest and a source checkpoint
uv run .pycandle/scripts/weight_extractor.py \
    --checkpoint path/to/large_model.bin \
    --manifest .pycandle/traces/my_large_model_meta_manifest.json \
    --out .pycandle/traces/my_large_model_weights.safetensors
```

This script will:
1. Read the manifest to know exactly which parameters are needed.
2. Load the checkpoint (streams if safetensors, CPU-load if pickle).
3. Filter and save only the required tensors to a new, optimized file.

## Benefits
- **Zero Memory Consumption**: Tracing happens in milliseconds with no RAM pressure.
- **Portability**: The `.pycandle` folder contains everything needed to reproduce the trace.
- **Power User DX**: Separates architecture capture from weight management.
