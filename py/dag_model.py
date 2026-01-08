import torch
import torch.nn as nn
from spy import GoldenRecorder

class ComplexModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(16, 32, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(32 * 10, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x is (B, 32, 10)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = ComplexModel()
    recorder = GoldenRecorder()
    
    # Dummy input (B, C, T)
    x = torch.randn(1, 16, 10)
    
    recorder.record(model, x)
    recorder.trace_fx(model, x)
    recorder.save("complex_model", use_fx=True)
