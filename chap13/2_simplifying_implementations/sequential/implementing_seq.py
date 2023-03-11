import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(4,16),
    nn.ReLU(),
    nn.Linear(16, 32),
    nn.ReLU()
)

print(model)