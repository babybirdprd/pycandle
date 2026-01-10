import torch
import sys
import os

# Ensure we can import the local spy module
# In the .pycandle/scripts environment, spy.py is right next to us.
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from spy import GoldenRecorder

# TODO: Import your model here
# from src.model import MyModel

def main():
    print("🚀 Starting PyCandle Recorder...")
    
    # 1. Initialize Model
    # model = MyModel()
    # model.eval()

    # 2. Create Recorder
    recorder = GoldenRecorder(output_dir="../traces")

    # 3. Create Dummy Inputs
    # dummy_input = torch.randn(1, 10, 32)
    
    # 4. Record
    # recorder.record(model, dummy_input)
    
    # 5. Save Record
    # recorder.save("my_model")
    
    print("✅ Recording complete!")

if __name__ == "__main__":
    main()
