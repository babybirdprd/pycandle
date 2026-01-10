import os
import sys
import json
import torch
from safetensors.torch import save_file
from typing import Dict, Any, Set

def extract_weights(checkpoint_path: str, manifest_path: str, output_path: str, mapper_path: str = None):
    """
    Extracts only the weights specified in the manifest from a PyTorch checkpoint.
    Supports .bin/.pt (Pickle) and .safetensors.
    """
    if not os.path.exists(manifest_path):
        print(f"❌ Manifest not found: {manifest_path}")
        sys.exit(1)
        
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Identify active parameters
    active_params: Set[str] = set()
    for name, meta in manifest.items():
        if name.startswith('_'): 
            continue
        if 'parameters' in meta:
            for p in meta['parameters']:
                active_params.add(f"{name}.{p}")
    
    print(f"🔍 Manifest contains {len(active_params)} active parameters.")
    
    # Load weights selectively
    weights: Dict[str, torch.Tensor] = {}
    
    if checkpoint_path.endswith('.safetensors'):
        from safetensors import safe_open
        with safe_open(checkpoint_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                if key in active_params:
                    weights[key] = f.get_tensor(key)
    else:
        # Pickle-based (.bin, .pt, .pth)
        # We use weights_only=True for security and map_location='cpu' for memory
        try:
            state_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
            for k, v in state_dict.items():
                if k in active_params:
                    weights[k] = v
        except Exception as e:
            print(f"❌ Failed to load checkpoint: {e}")
            # Fallback for older torch versions or complex pickles
            state_dict = torch.load(checkpoint_path, map_location='cpu')
            for k, v in state_dict.items():
                if k in active_params:
                    weights[k] = v

    if not weights:
        print("⚠️ No matching weights found in checkpoint!")
        # Print a few examples from the checkpoint if possible
        return

    # Optional renaming
    if mapper_path and os.path.exists(mapper_path):
        import re
        with open(mapper_path, 'r') as f:
            mappings = json.load(f)
        
        print(f"🔄 Applying {len(mappings)} renaming patterns...")
        renamed_weights = {}
        # Sort mappings by length of pattern (desc) or just alphabetical for consistency?
        # Typically we want deterministic order.
        sorted_patterns = sorted(mappings.items())
        
        for k, v in weights.items():
            new_k = k
            for pattern, replacement in sorted_patterns:
                new_k = re.sub(pattern, replacement, new_k)
            renamed_weights[new_k] = v
        weights = renamed_weights

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    
    save_file(weights, output_path)
    print(f"✅ Surgically extracted {len(weights)} weights to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Surgically extract model weights.")
    parser.add_argument("--checkpoint", required=True, help="Path to PyTorch checkpoint (.bin, .pt, .safetensors)")
    parser.add_argument("--manifest", required=True, help="Path to manifest.json")
    parser.add_argument("--out", required=True, help="Output .safetensors path")
    parser.add_argument("--map", help="Optional JSON mapping file for renaming")
    args = parser.parse_args()
    
    extract_weights(args.checkpoint, args.manifest, args.out, args.map)
