from torch import nn


class TimeDelayNeuralNetwork(nn.Module):
    def __init__(self, neurons_first_layer=5, neurons_second_layer=10):
        super().__init__()
        self.architecture = nn.Sequential(
            nn.Linear(neurons_first_layer, neurons_second_layer),
            nn.Threshold(0.5, 0),
            nn.Linear(neurons_second_layer, 1)
        )

    def forward(self, x):
        return self.architecture(x)
