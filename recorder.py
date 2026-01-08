import torch
import sys
import os

# Try to import pycandle spy. 
# in a real setup this would be installed, but for now we look in relative paths common in this workspace.
try:
    from pycandle.spy import GoldenRecorder
except ImportError:
    # Add potential fallback paths
    possible_paths = ["py", "../py", "../../py"]
    for p in possible_paths:
        if os.path.exists(os.path.join(p, "spy.py")):
            sys.path.append(p)
            break
    from spy import GoldenRecorder

# TODO: Import your model class
# from my_project.model import MyModel

def main():
    print("🚀 Initializing model configuration...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"   Device: {device}")

    # TODO: Instantiate your model
    # model = MyModel().to(device)
    # model.eval()

    # TODO: Create dummy input matching your model's requirement
    # dummy_input = torch.randn(1, 3, 224, 224).to(device)

    print("🎥 Starting recording...")
    recorder = GoldenRecorder(output_dir="traces")
    
    # TODO: Run the forward pass with the recorder
    # recorder.record(model, dummy_input)
    
    # Save the trace
    name = "debug_run"
    recorder.save(name)
    print(f"✅ Recording saved to traces/{name}")

if __name__ == "__main__":
    main()
