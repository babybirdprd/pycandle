import torch
import torch.nn as nn
from spy import GoldenRecorder

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(16 * 10, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = SmallModel()
    recorder = GoldenRecorder()
    
    # Dummy input (B, C, T)
    x = torch.randn(1, 1, 10)
    
    recorder.record(model, x)
    recorder.save("small_model")
