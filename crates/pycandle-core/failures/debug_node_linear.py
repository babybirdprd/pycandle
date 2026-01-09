
import torch
from safetensors.torch import load_file
import matplotlib.pyplot as plt
import numpy as np

def analyze():
    print(f"🔍 Analyzing Failure: {'node_linear'}")
    tensors = load_file("node_linear.safetensors")
    rust = tensors["rust_actual"]
    gold = tensors["py_golden"]

    diff = (rust - gold).abs()
    print(f"  Max Diff: {diff.max().item():.6f}")
    print(f"  MSE:      {(diff ** 2).mean().item():.8f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Rust Tensor Histogram")
    plt.hist(rust.flatten().float().numpy(), bins=50, alpha=0.7, label='Rust')
    plt.hist(gold.flatten().float().numpy(), bins=50, alpha=0.7, label='Gold')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Difference Heatmap (First Slice)")
    if diff.ndim > 1:
        plt.imshow(diff.flatten(0, -2)[0].float().numpy(), cmap='hot', aspect='auto')
    else:
        plt.plot(diff.float().numpy())
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyze()
