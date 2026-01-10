import torch
import torch.fx as fx
import onnx
from onnx2torch import convert
import json
import os
import argparse
from safetensors.torch import save_file
from spy import GoldenRecorder

def convert_onnx(onnx_path, name, out_dir):
    print(f"Loading ONNX model from {onnx_path}...")
    # Convert ONNX directly to a PyTorch Module (GraphModule)
    try:
        model = convert(onnx_path)
    except Exception as e:
        print(f"❌ Failed to load/convert ONNX model: {e}")
        return

    print("✅ ONNX loaded as PyTorch module.")
    
    # We need dummy input to trace it. 
    # Attempt to infer input shape from ONNX graph?
    # For now, let's create a dummy input based on heuristics or require user Arg?
    # onnx2torch usually handles dynamic shapes, but GoldenRecorder needs concrete tensors for shape recording.
    
    # Inspect ONNX graph for input shapes
    onnx_model = onnx.load(onnx_path)
    input_shapes = []
    
    # Very basic shape inference
    for input in onnx_model.graph.input:
        shape = []
        for d in input.type.tensor_type.shape.dim:
            if d.dim_value > 0:
                shape.append(d.dim_value)
            else:
                shape.append(1) # Default dynamic dim to 1
        input_shapes.append(shape)
    
    print(f" inferred input shapes: {input_shapes}")
    
    dummy_inputs = []
    for shape in input_shapes:
        dummy_inputs.append(torch.randn(*shape))
        
    if not dummy_inputs:
        print("⚠️ Could not infer inputs, using default (1, 10)")
        dummy_inputs = [torch.randn(1, 10)]

    print("🎥 Recording trace...")
    recorder = GoldenRecorder(output_dir=out_dir)
    
    # Record
    # We might need to handle tuple inputs if there are multiple
    if len(dummy_inputs) == 1:
        recorder.record(model, dummy_inputs[0])
    else:
        recorder.record(model, *dummy_inputs)
        
    recorder.save(name)
    print(f"✅ Converted and saved to {out_dir}/{name}_manifest.json")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    convert_onnx(args.onnx, args.name, args.out)
