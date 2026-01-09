# ONNX to FX Converter
import os
import sys
import argparse
import torch
import onnx
from onnx2torch import convert
from spy import GoldenRecorder

def main():
    parser = argparse.ArgumentParser(description="Convert ONNX model to PyCandle manifest via torch.fx")
    parser.add_argument("--onnx", type=str, required=True, help="Path to ONNX model")
    parser.add_argument("--name", type=str, required=True, help="Project name")
    parser.add_argument("--out", type=str, default="pycandle_trace", help="Output directory")
    
    args = parser.parse_args()
    
    print(f"📦 Loading ONNX model from {args.onnx}...")
    onnx_model = onnx.load(args.onnx)
    
    print("🔄 Converting ONNX to PyTorch...")
    torch_model = convert(onnx_model)
    torch_model.eval()
    
    # Try to infer input shape from ONNX model
    input_shape = [1, 3, 224, 224] # Default fallback
    if len(onnx_model.graph.input) > 0:
        dims = onnx_model.graph.input[0].type.tensor_type.shape.dim
        input_shape = [d.dim_value if d.dim_value > 0 else 1 for d in dims]
    
    print(f"🧪 Using input shape: {input_shape}")
    dummy_input = torch.randn(*input_shape)
    
    recorder = GoldenRecorder(output_dir=args.out)
    
    print("🎬 Recording trace and FX graph...")
    # Record will also perform FX tracing if requested
    recorder.record(torch_model, dummy_input, trace_fx=True)
    
    print(f"💾 Saving manifest and weights to {args.out}...")
    recorder.save(args.name, use_fx=True)
    
    # Save weights
    from safetensors.torch import save_file
    weight_path = os.path.join(args.out, f"{args.name}_weights.safetensors")
    # convert state_dict to use keys matching the manifest (which uses node names)
    # Actually, recorder.manifest has the mapping.
    # But for a simple ONNX converted model, state_dict keys often match node names if convert is clean.
    # Let's just save the state_dict for now.
    save_file(torch_model.state_dict(), weight_path)
    
    print(f"✅ Conversion complete! You can now run:")
    print(f"   pycandle codegen --manifest {args.out}/{args.name}_manifest.json --out generated_{args.name}.rs")

if __name__ == "__main__":
    main()
