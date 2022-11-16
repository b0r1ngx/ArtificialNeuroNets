import torch
from torch import nn

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Used device: {device}")
torch.device(device)


class TimeDelayNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(5, 10),
            nn.Threshold(0.5, 0),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        pass


model = TimeDelayNeuralNetwork()
print(model)

def ARn():
    pass
