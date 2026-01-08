import torch
import torch.nn as nn
from spy import GoldenRecorder

class HintsModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Ambiguous: Both 64
        self.embedding = nn.Embedding(64, 64)
        self.linear = nn.Linear(64, 64)

    def forward(self, x):
        x = self.embedding(x)
        x = self.linear(x)
        return x

model = HintsModel()
recorder = GoldenRecorder(output_dir="test_trace_hints")
x = torch.randint(0, 64, (1, 10))
recorder.record(model, x)

# Provide hints to resolve ambiguity
hints = {
    "vocab_size": 64,
    "hidden_dim": 64
}

recorder.save("hints_test", hints=hints)
