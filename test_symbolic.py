import torch
import torch.nn as nn
from spy import GoldenRecorder

class SmallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(50257, 768)
        self.linear1 = nn.Linear(768, 2048)
        self.linear2 = nn.Linear(2048, 768)
        self.ln = nn.LayerNorm(768)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        x = self.ln(x)
        return x

model = SmallModel()
recorder = GoldenRecorder(output_dir="test_trace")
# Embedding input: (batch, seq)
x = torch.randint(0, 50257, (1, 10))
recorder.record(model, x)
recorder.save("symbolic_test")
